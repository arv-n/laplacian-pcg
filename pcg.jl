### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ aa0f9f7e-6409-11eb-3125-1301a8b20ef9
using PlutoUI,PyPlot,SparseArrays,
IterativeSolvers,Preconditioners,
LinearAlgebra,Triangulate,IncompleteLU,Arpack

# ╔═╡ d515e2f4-640a-11eb-1a73-159e6d21533e
html"<button onclick='present()'>present</button>"

# ╔═╡ 3a4fc858-647c-11eb-0a23-e7c8ec8348c6
#plot solution
function plot(u,pointlist, trianglelist)
    cmap="coolwarm"
    levels=6
    x=pointlist[1,:]
    y=pointlist[2,:]
    ax=PyPlot.matplotlib.pyplot.gca()
    ax.set_aspect(1)
    #  tricontour/tricontourf takes triangulation as argument
    PyPlot.tricontourf(x,y,triangles=trianglelist,u,level=levels,cmap=cmap)
	colorbar(orientation="horizontal",spacing="proportional",
		ticks=LinRange(minimum(u),maximum(u),3),shrink=0.75,pad=0.05)
    PyPlot.tricontour(x,y,triangles=trianglelist,u,levels=levels,colors="k")
	PyPlot.xticks(visible=false);
	PyPlot.yticks(visible=false);	
end

# ╔═╡ 2c2f9998-647b-11eb-3ddc-11f90d029e3c
#plot grid and solution
function plotpair(pointlist,trianglelist,u=zeros(size(pointlist,2)))
	PyPlot.clf()
	PyPlot.subplot(121)
	PyPlot.title("Grid")
	PyPlot.triplot(pointlist[1,:],pointlist[2,:],triangles=trianglelist)
	PyPlot.yticks(LinRange(-1,1,5));
	ax1=PyPlot.matplotlib.pyplot.gca()
	ax1.set_aspect(1)
	PyPlot.margins(0)
	PyPlot.subplot(122)
	PyPlot.title("Solution")
	plot(u,pointlist,trianglelist)
	gcf();
end

# ╔═╡ 12755762-688c-11eb-14a3-479950c1a3a1
#plot convergence history
function plotconv(cht::Array{IterativeSolvers.ConvergenceHistory{true,Nothing}},
		labels::Array{String,1})
	clf();
	for i in 1:length(cht)
		resnorm = cht[i][:resnorm]
		PyPlot.plot(1:length(resnorm),resnorm,"-X",label=labels[i],lw=0.9,mew=0.01)
	end
	ax=PyPlot.matplotlib.pyplot.gca()
	PyPlot.xlim(0,50)
	ax.set_xlabel("PCG Iteration")
	ax.set_ylabel("Residual Norm (Preconditioned)")
	ax.set_yscale("log")
	PyPlot.grid()
	legend()
	gcf()
end

# ╔═╡ ed36e1ac-696f-11eb-00b8-5d6cc89a731e
#plot eigenvals in increasing order
function plotspectra(λ::Array{Array{Float64,1}},labels::Array{String,1})
	clf();
	for i in 1:length(λ)
		PyPlot.plot(λ[i],label=labels[i]);
	end
	ax=PyPlot.matplotlib.pyplot.gca()
	ax.set_xlabel("λ_i")
	ax.set_yscale("log")
	legend()
	gcf()
end

# ╔═╡ cd77adb2-64bd-11eb-1700-8dc2f703553d
begin
	mutable struct triMesh
		pointlist::Array{Float64} 
	    trianglelist::Array{Int64} 
		intpoints::Array{Bool}
		nelem::Int64 
		dofs::Int64 #dofs
		np::Int64
	end
	triMesh(;dofs=3) = triMesh([],[],[],0,dofs,0)
end

# ╔═╡ 8852442c-6481-11eb-281c-e53ac9aa06f3
function generate_tensor_mesh(meshObj::triMesh)
	
	#assuming equal gridpoints along both directions
	N = isqrt(meshObj.dofs)
	
	nx = N + 2;
	ny = N + 2; 
	xx=LinRange(-1,1,nx);
	yy=LinRange(-1,1,ny);
	
	#generate points
	x = reshape(repeat(xx,length(yy)),nx,ny);
	y = reshape(repeat(yy,length(xx)),ny,nx)';
	
	pointlist = Matrix{Float64}([x[:] y[:]]')
	
	#generate connectivities
	ix = reshape(1:(nx*ny),nx,ny);
	ix1 = ix[1:nx-1,1:ny-1]; ix1=ix1[:];
	ix2= ix[2:nx   ,1:ny-1]; ix2=ix2[:];
	ix3= ix[1:nx-1 ,2:ny]; ix3= ix3[:];
	ix4= ix[2:nx   ,2:ny]; ix4=ix4[:];
	
	trianglelist = [ix1 ix2 ix3; ix2 ix4 ix3];
	trianglelist = trianglelist.-1;
	
	meshObj.intpoints = .!((pointlist[1,:] .== -1.0) .| 
			(pointlist[1,:] .== 1.0) .| (pointlist[2,:] .== -1.0) 
			.| (pointlist[2,:] .== 1.0));
	
	meshObj.nelem = size(trianglelist,1); meshObj.np = size(pointlist,2);
	meshObj.pointlist = pointlist; meshObj.trianglelist = trianglelist; 
end

# ╔═╡ 1c2c594a-69ef-11eb-0419-3bca2ee1d692
N=81;

# ╔═╡ 1b8a3096-64c4-11eb-21c6-6b0b43f7e84d
e1 = triMesh(dofs=N);

# ╔═╡ 383a3b3e-66e8-11eb-1594-5934cb8587c5
k(x::Float64,y::Float64) = 161.45*((x<=0.) & (y<=0.)) + (1)*((x>0.) & (y<=0.)) + 161.45*((x>0.) & (y>0.)) + 1*((x<=0.) & (y>0.))
#coefficient

# ╔═╡ 1480af62-64ad-11eb-28a7-d34225155b25
generate_tensor_mesh(e1);

# ╔═╡ 79af845a-64d8-11eb-3458-1f654490b456
function generatetransformation2D(e,e2,x,y)
    dx1 = x[e2[e,2]] - x[e2[e,1]];
    dy1 = y[e2[e,2]] - y[e2[e,1]];

    dx2 = x[e2[e,3]] - x[e2[e,1]];
    dy2 = y[e2[e,3]] - y[e2[e,1]];

# determinant on each triangle
    Fdet = dx1 * dy2 - dx2 * dy1;

# transformation jacobian on each triangle
    Finv = zeros(2, 2);
    Finv[1,1] =  dy2 / Fdet ;
    Finv[1,2] = -dx2 / Fdet ;
    Finv[2,1] = -dy1 / Fdet ;
    Finv[2,2] =  dx1 / Fdet ;
    return Fdet, Finv
end

# ╔═╡ f906340a-64f5-11eb-2dd3-7110e723d06c
function localstiff2D(Fdet, Finv)
    gradphi=[-1 -1;1 0;0 1]
    dphi    = gradphi * Finv;
    sloc = 1 / 2 * (dphi[:,1] * dphi[:,1]' + dphi[:,2] * dphi[:,2]') * Fdet;
    return sloc
end

# ╔═╡ 8a1f0418-64f7-11eb-00bc-27f5c923357c
function localmass2D(Fdet)
    mloc = Fdet * [1 1 / 2 1 / 2;1 / 2 1 1 / 2;1 / 2 1 / 2 1] / 12;
    return mloc
end

# ╔═╡ 48c11296-64cc-11eb-12e6-c95142efa7c1
function global_assemble(meshObj::triMesh)
	
	nphi = 3;
	ne = meshObj.nelem;	
	e2 = meshObj.trianglelist .+ 1;
	x = meshObj.pointlist[1,:];
	y = meshObj.pointlist[2,:];
	
    ## build matrices
    ii = zeros(Int64, ne, nphi, nphi); # sparse i-index
    jj = zeros(Int64, ne, nphi, nphi); # sparse j-index
    aa = zeros(ne, nphi, nphi); # entry of Galerkin matrix
    bb = zeros(ne, nphi, nphi); # entry in mass-matrix (to build rhs)
	cc = zeros(ne, nphi, nphi); # entry of Laplacian matrix
	
	for e = 1:ne 
		Fdet, Finv = generatetransformation2D(e,e2,x,y);
		
		#calculate coefficient at the centroid of each element
		x_c = (x[e2[e,1]]+x[e2[e,2]]+x[e2[e,3]])/3.;
		y_c = (y[e2[e,1]]+y[e2[e,2]]+y[e2[e,3]])/3.;	
		ke = k(x_c,y_c);
		
		# build local matrices (mass, stiffness, ...)
        sloc = ke*localstiff2D(Fdet, Finv); # element stiffness matrix
		lloc = localstiff2D(Fdet, Finv); #element laplacian matrix
        mloc = localmass2D(Fdet);       # element mass matrix
        
        # compute i,j indices of the global matrix
        dofs = e2[e,:];
		
		# compute a(i,j) values of the global matrix
        for i = 1:nphi
            for j = 1:nphi
                ii[e,i,j] = dofs[i]; # local-to-global
                jj[e,i,j] = dofs[j]; # local-to-global
                aa[e,i,j] = sloc[i,j]; #Stiffness matrix
				cc[e,i,j] = lloc[i,j]; #Laplacian matrix
                bb[e,i,j] = mloc[i,j]; #mass matrix
            end
        end
		
	end
	# create sparse matrices
	A = sparse(ii[:], jj[:], aa[:]);
	L = sparse(ii[:], jj[:], cc[:]);
	M = sparse(ii[:], jj[:], bb[:]);

	# build rhs 
	rhs = M*ones(meshObj.np, 1);	
	
	int = meshObj.intpoints;
	
	return A[int,int],L[int,int],rhs[int];
	
end

# ╔═╡ ddd54870-673a-11eb-01a5-ffe24cc7a2ad
A,L,rhs = global_assemble(e1);

# ╔═╡ 3309239c-6723-11eb-01ac-ab241fb96a42
function solve_fem(meshObj,A,L,rhs,pcg="default")	 
	u = zeros(meshObj.np); 
	int = meshObj.intpoints
	if(pcg=="cg")
		u[int],ch=cg(A,rhs,log=true)
	elseif(pcg=="diagonal")
		p = DiagonalPreconditioner(A)
		u[int],ch=cg(A,rhs,Pl=p,log=true)
	elseif(pcg=="ilu")
		p = ilu(A,τ=1e-2)
		u[int],ch=cg(A,rhs,Pl=p,log=true)
	elseif(pcg=="laplacian")
		p = lu(L)
		u[int],ch=cg(A,rhs,Pl=p,log=true)
	end
	return u,ch
end

# ╔═╡ 89f1d0aa-6723-11eb-1b3b-e5a761f2fa18
u,ch = solve_fem(e1,A,L,rhs,"cg");

# ╔═╡ ea11ad48-6723-11eb-25fe-7116c904b723
pl = e1.pointlist; tl = e1.trianglelist;

# ╔═╡ 6ae5e60a-64f9-11eb-242a-c9e6bf68f24c
plotpair(pl,tl,u)

# ╔═╡ abff15fe-6890-11eb-27ca-258853f71f8c
_,ch2 = solve_fem(e1,A,L,rhs,"ilu");

# ╔═╡ bf263a14-6957-11eb-1746-1741c11adf4d
__,ch3 = solve_fem(e1,A,L,rhs,"laplacian");

# ╔═╡ c09b2a4a-6896-11eb-0289-cb57389c9a32
plotconv([ch,ch2,ch3],["Diagonal","ilu","Laplace"])

# ╔═╡ a4bbc224-6949-11eb-3e01-89f56bbf93a4
Ls=sqrt(Matrix(L));

# ╔═╡ 1ef89e6c-6969-11eb-22ca-bf8042ac79e0
Linv=inv(Ls);

# ╔═╡ 0059f2e4-692d-11eb-2f18-1987e20dc34b
AL=Linv*A*Linv;

# ╔═╡ 21fcaac2-692d-11eb-38ca-0be1fae1718e
cond(AL)

# ╔═╡ ef26302c-68a4-11eb-29d7-730bda510b12
λ_A,=eigs(A,nev=81,which=:SM);

# ╔═╡ fc555ece-6970-11eb-01ea-c99033916c7d
λ_L,=eigs(L,nev=81,which=:SM);

# ╔═╡ ae8804d4-6971-11eb-0cbf-cfc0b717b388
begin
	λ_AL,v_AL=eigen(AL);
	λ_AL=sort(real.(λ_AL));
	v_AL=real.(v_AL);
	v_AL = v_AL[:, sortperm(λ_AL)];
	"Eigenvalues and eigenvectors of AL calculated"
end

# ╔═╡ 7cff19fe-696f-11eb-27e0-eb6fb8451d61
plotspectra([λ_A,λ_L,λ_AL],["A","L","AL"])

# ╔═╡ e75ba2a0-6982-11eb-0145-6f79898b4d43
v̄L=zeros(N,N);ωL=zeros(N);

# ╔═╡ 96916bc4-6983-11eb-2c9b-93ff759b8f02
qL=Linv*rhs/norm(Linv*rhs);

# ╔═╡ 4c27db96-6982-11eb-32f4-4d27af784835
for i = 1:N
	v_AL[:,i]=real(v_AL[:,i]);
	v̄L[:,i]=v_AL[:,i];
	ωL[i] = dot(v̄L[:,i],qL)^2;
end

# ╔═╡ cad53db6-6983-11eb-3500-c10549dcdd48
clf();PyPlot.plot(λ_AL,sort(ωL),"-x");PyPlot.matplotlib.pyplot.gca().set_yscale("log");gcf()

# ╔═╡ cfd25188-69ff-11eb-3ed8-bd7044663058
#TODOS: - k(x,y) vs Eigenvals for the 4 examples
# - mendelsohn-pairing 
# - ichol implementation
# - writing: (prelim outline: -intro-literature-motivatingexample-proof-resultingegs-explainations-conclusions
# - cosmetic changes

# ╔═╡ Cell order:
# ╟─d515e2f4-640a-11eb-1a73-159e6d21533e
# ╠═aa0f9f7e-6409-11eb-3125-1301a8b20ef9
# ╟─3a4fc858-647c-11eb-0a23-e7c8ec8348c6
# ╟─2c2f9998-647b-11eb-3ddc-11f90d029e3c
# ╟─12755762-688c-11eb-14a3-479950c1a3a1
# ╟─ed36e1ac-696f-11eb-00b8-5d6cc89a731e
# ╟─8852442c-6481-11eb-281c-e53ac9aa06f3
# ╠═cd77adb2-64bd-11eb-1700-8dc2f703553d
# ╠═1c2c594a-69ef-11eb-0419-3bca2ee1d692
# ╠═1b8a3096-64c4-11eb-21c6-6b0b43f7e84d
# ╠═383a3b3e-66e8-11eb-1594-5934cb8587c5
# ╠═1480af62-64ad-11eb-28a7-d34225155b25
# ╟─79af845a-64d8-11eb-3458-1f654490b456
# ╟─f906340a-64f5-11eb-2dd3-7110e723d06c
# ╟─8a1f0418-64f7-11eb-00bc-27f5c923357c
# ╟─48c11296-64cc-11eb-12e6-c95142efa7c1
# ╠═ddd54870-673a-11eb-01a5-ffe24cc7a2ad
# ╠═3309239c-6723-11eb-01ac-ab241fb96a42
# ╠═89f1d0aa-6723-11eb-1b3b-e5a761f2fa18
# ╠═ea11ad48-6723-11eb-25fe-7116c904b723
# ╠═6ae5e60a-64f9-11eb-242a-c9e6bf68f24c
# ╠═abff15fe-6890-11eb-27ca-258853f71f8c
# ╠═bf263a14-6957-11eb-1746-1741c11adf4d
# ╠═c09b2a4a-6896-11eb-0289-cb57389c9a32
# ╠═a4bbc224-6949-11eb-3e01-89f56bbf93a4
# ╠═1ef89e6c-6969-11eb-22ca-bf8042ac79e0
# ╠═0059f2e4-692d-11eb-2f18-1987e20dc34b
# ╠═21fcaac2-692d-11eb-38ca-0be1fae1718e
# ╠═ef26302c-68a4-11eb-29d7-730bda510b12
# ╠═fc555ece-6970-11eb-01ea-c99033916c7d
# ╟─ae8804d4-6971-11eb-0cbf-cfc0b717b388
# ╠═7cff19fe-696f-11eb-27e0-eb6fb8451d61
# ╠═e75ba2a0-6982-11eb-0145-6f79898b4d43
# ╠═96916bc4-6983-11eb-2c9b-93ff759b8f02
# ╠═4c27db96-6982-11eb-32f4-4d27af784835
# ╠═cad53db6-6983-11eb-3500-c10549dcdd48
# ╠═cfd25188-69ff-11eb-3ed8-bd7044663058
