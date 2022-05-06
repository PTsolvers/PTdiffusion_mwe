# julia --project -O3 --check-bounds=no diffusion_2D_impl_xpu.jl
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qTx, qTy, T, Lam, dx, dy)
    @all(qTx) = -@av_xi(Lam)*@d_xi(T)/dx
    @all(qTy) = -@av_yi(Lam)*@d_yi(T)/dy
    return
end

macro dtau() esc(:( 1.0/(1.0/(min_dxy2/(@inn(Lam)/@inn(ρCp))/4.1) + 1.0/dt) )) end
@parallel function compute_update!(T, dTdt, Told, qTx, qTy, Lam, ρCp, dt, min_dxy2, dmp, dx, dy)
    @inn(dTdt) = dmp*@inn(dTdt) + ( -(@inn(T) - @inn(Told))/dt - 1.0/@inn(ρCp)*(@d_xa(qTx)/dx + @d_ya(qTy)/dy) )
    @inn(T)    = @inn(T) + @dtau()*@inn(dTdt)
    return
end

@parallel function compute_residual!(ResT, T, Told, qTx, qTy, ρCp, dt, dx, dy)
    @all(ResT) = -(@inn(T) - @inn(Told))/dt - 1.0/@inn(ρCp)*(@d_xa(qTx)/dx + @d_ya(qTy)/dy)
    return
end

@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly   = 10.0, 10.0    # domain extend
    λ0       = 0.5           # background heat conductivity
    ttot     = 0.5           # total time
    # Derived physics
    dt       = ttot/2        # time step, if explicit: min_dxy2/maximum(Lam./ρCp)/4.1
    # Numerics
    n        = 2
    nx, ny   = n*32, n*32    # number of grid points
    nout     = 50            # sparse check
    dmp      = 0.8           # solver acceleration
    iterMax  = 1e4           # max allowed iters
    tol      = 1e-8          # nonlinear tolerance
    # Derived numerics
    dx, dy   = Lx/nx, Ly/ny  # grid cell size
    xc, yc   = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    min_dxy2 = min(dx,dy)^2
    # Array initialisation
    T        = Data.Array(exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/2).^2))
    Told     = copy(T)
    qTx      = @zeros(nx-1,ny-2)
    qTy      = @zeros(nx-2,ny-1)
    dTdt     = @zeros(nx  ,ny  )
    ResT     = @zeros(nx-2,ny-2)
    ρCp      =   ones(nx  ,ny  )
    ρCp[((xc.-Lx/3).^2 .+ (yc'.-Ly/3).^2).<1.0].=0.01
    ρCp      = Data.Array(ρCp)
    Lam      = λ0 .+ 0.1.*@rand(nx,ny)
    time=0.0; it=0; t_tic=0.0; niter=0; nitertot=0
    # Time loop
    while time < ttot
        it += 1
        err = 2*tol; iter = 0
        while err>tol && iter<iterMax
            iter += 1
            if (it == 1 && iter == 11) t_tic = Base.time(); niter = 0 end
            @parallel compute_flux!(qTx, qTy, T, Lam, dx, dy)
            @parallel compute_update!(T, dTdt, Told, qTx, qTy, Lam, ρCp, dt, min_dxy2, dmp, dx, dy)
            nitertot += 1; niter += 1
            if (iter % nout == 0)
                @parallel compute_residual!(ResT, T, Told, qTx, qTy, ρCp, dt, dx, dy)
                err = norm(ResT)/length(ResT)
            end
        end
        time += dt
        if do_visu
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), #=clims=(0.0, 1.0),=# c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, Array(T)'; opts...))
        end
    end
    t_toc = Base.time() - t_tic
    @printf("Computed %d steps (total iterations = %d), physical time = %1.3f\n", it, nitertot, time)
    A_eff = 4/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                  # Execution time per iteration [s]
    T_eff = A_eff/t_it                   # Effective memory throughput [GB/s]
    @printf("Perf: time = %1.3f sec, T_eff = %1.2f GB/s\n", t_toc, round(T_eff, sigdigits=3))
    return
end

diffusion_2D(; do_visu=true)
