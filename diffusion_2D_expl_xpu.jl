# julia --project -O3 --check-bounds=no diffusion_2D_expl_xpu.jl
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf

@parallel function compute_flux!(qTx, qTy, T, Lam, dx, dy)
    @all(qTx) = -@av_xi(Lam)*@d_xi(T)/dx
    @all(qTy) = -@av_yi(Lam)*@d_yi(T)/dy
    return
end

@parallel function compute_update!(T, qTx, qTy, ρCp, dt, dx, dy)
    @inn(T) = @inn(T) - dt/@inn(ρCp)*(@d_xa(qTx)/dx + @d_ya(qTy)/dy)
    return
end

@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly   = 10.0, 10.0    # domain extend
    λ0       = 0.5           # background heat conductivity
    ttot     = 0.5           # total time
    # Numerics
    n        = 2
    nx, ny   = n*32, n*32    # number of grid points
    ndt      = 10            # sparse timestep computation
    nvis     = 50            # sparse visualisation
    # Derived numerics
    dx, dy   = Lx/nx, Ly/ny  # grid cell size
    xc, yc   = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    min_dxy2 = min(dx,dy)^2
    # Array initialisation
    T        = Data.Array(exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/2).^2))
    qTx      = @zeros(nx-1,ny-2)
    qTy      = @zeros(nx-2,ny-1)
    ρCp      =   ones(nx  ,ny  )
    ρCp[((xc.-Lx/3).^2 .+ (yc'.-Ly/3).^2).<1.0].=0.01
    ρCp      = Data.Array(ρCp)
    Lam      = λ0 .+ 0.1.*@rand(nx,ny)
    dt       = min_dxy2/maximum(Lam./ρCp)/4.1
    time=0.0; it=0; t_tic=0.0; niter=0
    # Time loop
    while time < ttot
        it += 1
        if (it == 11) t_tic = Base.time(); niter = 0 end
        if (it % ndt == 0) dt = min_dxy2/maximum(Lam./ρCp)/4.1 end # done every ndt to improve perf
        @parallel compute_flux!(qTx, qTy, T, Lam, dx, dy)
        @parallel compute_update!(T, qTx, qTy, ρCp, dt, dx, dy)
        niter += 1
        time += dt
        if do_visu && (it % nvis == 0)
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), #=clims=(0.0, 1.0),=# c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, Array(T)'; opts...))
        end
    end
    t_toc = Base.time() - t_tic
    @printf("Computed %d steps, physical time = %1.3f\n", it, time)
    A_eff = 4/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                  # Execution time per iteration [s]
    T_eff = A_eff/t_it                   # Effective memory throughput [GB/s]
    @printf("Perf: time = %1.3f sec, T_eff = %1.2f GB/s\n", t_toc, round(T_eff, sigdigits=3))
    return
end

diffusion_2D(; do_visu=true)
