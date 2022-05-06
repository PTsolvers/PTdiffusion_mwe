# mpiexecjl -n 1 julia --project diffusion_2D_expl_mxpu.jl
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using ImplicitGlobalGrid
import MPI
using Plots, Printf

maximum_g(A) = (sum_l = sum(A); MPI.Allreduce(sum_l, MPI.SUM, MPI.COMM_WORLD))

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
    n        = 1
    nx, ny   = n*32, n*32    # number of grid points
    me, dims = init_global_grid(nx, ny, 1) # MPI initialisation
    ndt      = 10            # sparse timestep computation
    # Derived numerics
    dx, dy   = Lx/nx_g(), Ly/ny_g()  # grid cell size
    min_dxy2 = min(dx,dy)^2
    # Array initialisation
    qTx      = @zeros(nx-1,ny-2)
    qTy      = @zeros(nx-2,ny-1)
    ρCp      =   ones(nx  ,ny  )
    rad2     =  zeros(nx  ,ny  )
    rad2    .= [((x_g(ix,dx,rad2)-0.5*Lx+dx/2)^2 + (y_g(iy,dy,rad2)-0.5*Ly+dy/2)^2) for ix=1:size(rad2,1), iy=1:size(rad2,2)]
    T        = Data.Array(exp.(.-rad2))
    ρCp[rad2.<1.0] .= 0.01
    ρCp      = Data.Array(ρCp)
    Lam      = λ0 .+ 0.1.*@rand(nx,ny)
    dt       = min_dxy2/maximum_g(Lam./ρCp)/4.1
    # Prepare visualisation
    if (me==0 && do_visu) ENV["GKSwstype"]="nul"; !ispath("./out") && mkdir("./out"); end
    nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
    T_v    = zeros(nx_v, ny_v) # global array for visu
    T_inn  = zeros(nx-2, ny-2) # no halo local array for visu
    xi_g, yi_g = dx+dx/2:dx:Lx-dx-dx/2, dy+dy/2:dy:Ly-dy-dy/2 # inner points only
    time=0.0; it=0; t_tic=0.0; niter=0
    # Time loop
    while time < ttot
        it += 1
        if (it == 11) t_tic = Base.time(); niter = 0 end
        if (it % ndt == 0) dt = min_dxy2/maximum_g(Lam./ρCp)/4.1 end # done every ndt to improve perf
        @parallel compute_flux!(qTx, qTy, T, Lam, dx, dy)
        @parallel compute_update!(T, qTx, qTy, ρCp, dt, dx, dy)
        update_halo!(T)
        niter += 1
        time += dt
    end
    t_toc = Base.time() - t_tic
    me==0 && @printf("Computed %d steps, physical time = %1.3f\n", it, time)
    A_eff = 4/1e9*nx_g()*ny_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                          # Execution time per iteration [s]
    T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]
    me==0 && @printf("Perf: time = %1.3f sec, T_eff = %1.2f GB/s\n", t_toc, round(T_eff, sigdigits=3))
    if do_visu
        T_inn .= Array(T[2:end-1,2:end-1]); gather!(T_inn, T_v)
        if me==0
            opts = (aspect_ratio=1, xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), #=clims=(0.0, 1.0),=# c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            heatmap(xi_g, yi_g, T_v'; opts...)
            savefig("./out/diff_2D.png")
        end
    end
    finalize_global_grid()
    return
end

diffusion_2D(; do_visu=true)
