### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

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

# ╔═╡ 85461a64-71b6-11eb-1153-01c5012c2d3d
md"""
## Laplacian Preconditioning of Elliptic PDEs: Localization of the Eigenvalues of the Discretized operator

Gergelits, T., Mardal, K. A., Nielsen, B. F., & Strakos, Z. (2019). SIAM Journal on Numerical Analysis.

Seminar **Numerical Linear Algebra, TU Berlin**


**Arvind Nayak, February 18 2021**
"""

# ╔═╡ ec9dea62-71b7-11eb-0c8d-ddbeaad47e00
md""" ## **Contents**
- Overview
- Motivation
- Elliptic PDEs and FEM
- The Conjugate Gradient Method
- Main Results
- Supporting numerical experiments
- Conclusion
"""

# ╔═╡ 8eae2aa4-71b8-11eb-1e08-eb530bbf3b44
md""" ## Introduction """

# ╔═╡ 9fc55114-71b8-11eb-256e-0fc33ea16a4d
md"""
- Analysis of Krylov subspace solvers for Hermitian matrices relies on their spectral properties.

- One seeks a preconditioner which yields parameter independent bounds for extreme eigenvalues. 

- *Operator preconditioning* is constructing preconditioners for "discrete linear operators arisen from a Galerkin approach" [^2]. Matching Galerkin discretizations of operators of complementary mapping properties.
"""

# ╔═╡ 829c6982-71b9-11eb-3e41-77e1d8d323f4
md"""Given a continuous bijective linear operator $A : V \mapsto W$ on function spaces $V$ and $W$, and another isomorphism $B : W \mapsto V$, then $BA$ will provide an endomorphism of $V$.[^2]""" 

# ╔═╡ 99b9c812-71bb-11eb-0511-972a2663b3a4
md"""The mapping properties between appropriate *Sobolev spaces* in order to derive a discrete preconditioner considered. [^3]""" 

# ╔═╡ 3a438aa0-71bc-11eb-1eac-df30d59b8104
md"""Bounds for the required number of Krylov subspace iterations can become independent of the mesh size and other important parameters."""

# ╔═╡ 797a74d2-71bb-11eb-0766-23e66d9a5c18
md"""## Motivation"""

# ╔═╡ 8cf8b7da-71bb-11eb-12ae-cb2d8054948f
md"""But what happens to convergence for problems with spatially variable coefficients?"""

# ╔═╡ 92281bfa-71bc-11eb-2dee-1907ead77773
md"""- The condition number estimate provided by *Operator Preconditioning* is of limited value.

- Convergence bounds based on single number do not capture adaption of Krylov subspace methods applied to non linear input data (the matrix and initial residual)

- Whole information of spectra needed to capture actual convergence behavior.

- Large outlying eigenvalues (or well separated clusters of large eigenvalues) can lead to *acceleration* of CG convergence. (assuming exact arithmetic)

- In practice, using finite precision computations, it can cause deterioration of the convergence rate."""

# ╔═╡ 6ea24578-71be-11eb-3a8d-33fe0014b0e8
md"""## Motivation"""

# ╔═╡ 71099b0e-71be-11eb-095f-2581ed5e01c3
md"""But what happens to convergence for problems with spatially variable coefficients?"""

# ╔═╡ 7bed953e-71be-11eb-0071-757715aab843
md"""- The condition number estimate provided by *Operator Preconditioning* is of limited value.

- Convergence bounds based on single number do not capture adaption of Krylov subspace methods applied to non linear input data (the matrix and initial residual)

- Whole information of spectra needed to capture actual convergence behavior.

- Large outlying eigenvalues (or well separated clusters of large eigenvalues) can lead to *acceleration* of CG convergence. (assuming exact arithmetic)

- In practice, using finite precision computations, it can cause deterioration of the convergence rate."""

# ╔═╡ 87d83dae-71be-11eb-1c98-e729954a844f
md"> We need a preconditioner that leads to a *favorable* distribution of the eigenvalues of preconditioned (Hermitian) matrix" 

# ╔═╡ db509706-71be-11eb-35e4-d5823c530c2a
md"## Mathematical Description"

# ╔═╡ 18f2455c-71bf-11eb-3360-81ef505644e4
md"""Consider a self-adjoint second order elliptic PDE
Consider a self-adjoint second order elliptic PDE in the form, where
```math
\begin{aligned}
& -\nabla\cdot(k(x)\nabla u) = f,
& &x \in \Omega \subset \mathbb{R}^d\\
&\hspace{1.5cm} u = 0, 
& &x \in \partial\Omega
\end{aligned}
```
and the corresponding generalized eigenvalue problem, 
```math
\begin{aligned}
& -\nabla\cdot(k(x)\nabla u) = \lambda\Delta u ,
& &x \in \Omega \\
&\hspace{1.5cm} u = 0, 
& &x \in \partial\Omega
\end{aligned}
```
We see that $f \in L^2(\Omega)$, and $k(x):\mathbb{R}^d \rightarrow \mathbb{R}$ \

$k(x)\geq \alpha > 0$ 
"""

# ╔═╡ cb8b1fdc-71c1-11eb-1c5a-79d4213317f0
md"""### Finite Element formulation"""

# ╔═╡ 7aed1ca2-71c0-11eb-34e9-9b8c3f1d3445
md"""Let $V_h \subset H_{0}^1(\Omega)$. We seek $u_h \in V_h$ such that, 
```math
\begin{aligned}
& \mathcal{A_h}u_h = f 
& & (\text{respectively} \mathcal{A_h}u_h = \lambda\mathcal{L_h}u_h)
\end{aligned}
```
where, discretization is via conforming FE method using Lagrange elements, leads to 
$\mathcal{A}_h,\mathcal{L}_h: V_h \rightarrow V_h^{\#}$
"""

# ╔═╡ 4520e0e2-71c3-11eb-19b0-9763903df14c
md"Finite dimensional subspace $V_h$ is spanned by piecewise polynomial basis functions $\phi_1,\dots,\phi_N$ with the local supports."

# ╔═╡ abae667e-71c3-11eb-368b-cfd9e5295b7c
md"""We then have $\mathcal{T}_i = \text{supp}(\phi_i)$, $i=1,\dots,N$"""

# ╔═╡ 194ae372-71c4-11eb-16a6-0dc3f89736e7
md"""Finally the matrix representations,

$$[A]_{i,j} = \int_{\Omega} \nabla\phi_i\cdot k\nabla\phi_j$$
$$[L]_{i,j} = \int_{\Omega} \nabla\phi_i\cdot \nabla\phi_j$$
$i,j = 1,\dots,N$.
"""

# ╔═╡ 96d1f4f2-71c4-11eb-140d-1382f39fe066
md"""## A bit about CG convergence"""

# ╔═╡ a1f53f76-71c4-11eb-112f-0f9c00f74390
md"""The energy norm of the error in CG can be given as, 

$$||x-x_j||_A^2= \min_{p\in\mathcal{P}_j}||p(A)(x-x_0)||^2_A$$, $j=1,2,...$"""

# ╔═╡ 76a94e90-71c5-11eb-352b-d1cc8edee31e
md"""Using the spectral decomposition of the matrix $A$ with eigenvalues $\lambda_1,\dots,\lambda_N$ with $v_1,\dots,v_N$ associated orthonormal eigenvectors,"""

# ╔═╡ bfefdf7e-71c5-11eb-0d02-89d589d49170
md"""We can rewrite it as, 

$$||x-x_j||_A^2= ||r_0||^2\sum_{l=1}^N \omega_l \frac{(p_j^{CG}(\lambda_l))^2}{\lambda_l}$$
with,
$r_0 = b-Ax_0$, $\omega_l = (z,v_l)^2$, $l=1,\dots,N$ and $z=r_0/||r_0||$"""

# ╔═╡ 27c70514-71c6-11eb-1703-2d77910f2002
md"""A simplified relation given by, 

$$\frac{||x-x_j||_A}{||x-x_0||_A}\leq 2 \left(\frac{\sqrt{\kappa(A)}-1}{\sqrt{\kappa(A)}+1}\right)^j$$
where $\kappa(A) = \lambda_N/\lambda_1$"""

# ╔═╡ 0a40431a-71c7-11eb-2807-bbaec4e2c21c
md""" Recall the limitations of this bound"""

# ╔═╡ 5c60b184-71c7-11eb-1673-37851c277024
md"# An Example"

# ╔═╡ ea4e69f6-71cb-11eb-379f-23f54f16a48f
md"""Let $k(x)$ be piecewise constant on the individual subdomains $\Omega_i$, $i=1,2,3,4$ $k_1, k_3 = 161.45$ and $k_2, k_4 = 1$."""

# ╔═╡ fdc1ed30-71c8-11eb-3f5c-3554de976606
begin
	sl_N = @bind n Slider(1:1:63, default=9);
	md"""
	dofs(N) = 1 $sl_N 3969
	"""
end

# ╔═╡ e66e4a5a-71c8-11eb-20d1-896fda01b251
md"# An Example"

# ╔═╡ 509b94ec-71ca-11eb-352c-d784033c5dd4
md"dofs(N) 1 $sl_N 3969"

# ╔═╡ e5cd2b6e-71d6-11eb-09e9-e38626e7efb7
md"""The condition number from the ICHOL factorization in MATLAB for the same problem was $\approx 1.16$"""

# ╔═╡ 305e5024-71cb-11eb-12b5-31da272983dc
md"## Let us plot the spectrum"

# ╔═╡ 4065df58-71cd-11eb-1250-d5b2c826e3cb
md"""The preconditioned system can be formulated as, 

$$A_L(L^{1/2}x) = L^{-1/2}b$$, where 

$$A_L = L^{-1/2}AL^{1/2}$$
"""

# ╔═╡ 9b351cb4-71cb-11eb-379a-eddd876a0c87
md"dofs(N) 1 $sl_N 3969"

# ╔═╡ 5b465b46-71cc-11eb-397d-85c3fcce10ac
md"""## Distribution functions"""

# ╔═╡ 030777b6-71cf-11eb-2dce-1f5c5f99b597
md"""Comparision between Laplace and ICHOL"""

# ╔═╡ 1452ebd4-71cf-11eb-3903-ef7234203ff4
md"N=49"

# ╔═╡ 24c08780-71cd-11eb-2f2f-e1bc01955371
md"![distfunc_N49](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/distfunc_N49.png)"

# ╔═╡ 1d240d4c-71cf-11eb-09b8-6983615a4992
md"N=3969"

# ╔═╡ c9201808-71ce-11eb-0d1e-af3bdb657f89
md"![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/dist_func_N%3D3969.png)"

# ╔═╡ 340f0494-71cf-11eb-0678-174d79dbef8c
md"""## Main Results"""

# ╔═╡ 42273880-71cf-11eb-25e8-77e494cc5a45
md""">Theorem 3.1: (pairing the eigenvalues and the intervals $k(\mathcal{T}_j)$,$ j=1,...,N$).
>>Let $\lambda_1,...\lambda_N$ be the eigenvalues of $L^{-1}A$ where A and L are the conforming FEM global stiffness and mass matrices respectively and let $k(x)$ be bounded and piecewise continuous. Then there exists a (possibly non unique) permutation $\pi$ such that the eigenvalues of the matrix $L^{-1}A$ satisfy, 
>> $$\lambda_{\pi} \in k(\mathcal{T}_j)$$"""

# ╔═╡ 6c91897e-71d0-11eb-1530-8b1fe21692e6
md""">Corollary 3.2: (pairing the eigenvalues and the nodal values).
>Using the same notation and assumptions from Theorem 3.1, consider any point $\hat{x}_j$ such that $\hat{x}_j \in \mathcal{T}_j$. Then the associated eigenvalue $\lambda_{\pi(j)}$ of the matrix $L^{-1}A$ 

$$|\lambda_{\pi(j)}-k(\hat{x}_j)| \leq \sup_{x \in \mathcal{T}_j}|k(x) - k(\hat{x}_j)|$$"""

# ╔═╡ b3bc1022-71d1-11eb-22f1-e7ff814684a0
md"""## Main Results"""

# ╔═╡ d9221816-71d1-11eb-1cf4-9166a9c456fd
md"""**Theorem 3.1**"""

# ╔═╡ bfa8ab34-71d1-11eb-0689-e1ef4f1552c2
md"![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/Thm3.1.png)"

# ╔═╡ e462e1ce-71d1-11eb-1747-9374db5a9384
md"""**Corollary 3.2**"""

# ╔═╡ f3dc1c24-71d1-11eb-0d0d-4d9a3e663f1c
md"![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/Corollary.png)"

# ╔═╡ 4f5e8b18-71d2-11eb-06bd-c1e048987ee7
md""" ## Numerical Results"""

# ╔═╡ 2e373cea-71f6-11eb-3930-09b0f1ea2c43
#TODO Implement the numerical results 

# ╔═╡ b227b602-71d2-11eb-3166-cb56652dfa5f
md"""**Illustration of Theorem 3.1**"""

# ╔═╡ fff70918-71d3-11eb-1c48-93ada3377cb6
md"""![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/k(x)nodalvalues.png)"""

# ╔═╡ 3e01ad42-71d4-11eb-1f92-8fbc59171e97
md""" ## Numerical Results"""

# ╔═╡ 9d34f122-71d4-11eb-22aa-05c099444e41
md"""**Illustration of Corollary 3.2**"""

# ╔═╡ ad40ceb8-71d4-11eb-1f65-fb1d231752b2
md"""![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/corollary32.png)"""

# ╔═╡ 764e8752-71d5-11eb-3cc0-85899265e2df
md"""## Explaination of the convergence"""

# ╔═╡ 99153286-71d5-11eb-114d-953047d72616
md"""![distfunc_N3969](https://raw.githubusercontent.com/arv-n/laplacian-pcg/main/explain.png)"""

# ╔═╡ edad8744-71d5-11eb-3ff2-e108636b319d
md"""# Conclusions"""

# ╔═╡ f5ff937e-71d5-11eb-3616-3b1e6ef88525
md""" - analysis of the operator $\mathcal{L^{-1}}\mathcal{A}$ generated by the preconditioning second order elliptic PDEs with the inverse of the Laplacian. A similar result holds for discrete case with matrices generated from conforming finite elements $L^{-1}A$.
- The eigenvalues of the matrix $L^{-1}A lie in resolution dependent intervals around the nodal values of the coefficient function k(x). 
- There exists a non unique pairing of the eigenvalues and the nodal values of k(x).
- Laplacian PCG utilizes this structure of the spectrum to accelerate the iterations."""

# ╔═╡ a194d4fe-71ba-11eb-273b-0fcefe96bcc0
md""" ## References
[^1]: Gergelits, T., Mardal, K. A., Nielsen, B. F., & Strakos, Z. (2019). Laplacian preconditioning of elliptic PDEs: Localization of the eigenvalues of the discretized operator. SIAM Journal on Numerical Analysis, 57(3), 1369-1394.
[^2]: Hiptmair, R. (2006). Operator preconditioning. Computers & Mathematics with Applications, 52(5), 699-706.
[^3]: B. F. Nielsen, A. Tveito, and W. Hackbusch, Preconditioning by inverting the Laplacian: An analysis of the eigenvalues, IMA J. Numer. Anal., 29 (2009)
[^4]: Liesen, J.,  Numerical Linear Algebra 1 Course Notes, TU Berlin (2019-20)
[^5]: M. R. Hestenes and E. Stiefel, Methods of conjugate gradients for solving linear systems, J. Res. Nat. Bur. Standards, 49 (1952)"""

# ╔═╡ aebfc210-71c9-11eb-08b5-f7aba9ac895f
N=n^2;

# ╔═╡ b56b808e-71d5-11eb-2f54-7f7c6eea5bbc
md"""# Notes and functions"""

# ╔═╡ 383a3b3e-66e8-11eb-1594-5934cb8587c5
k(x::Float64,y::Float64) = 161.45*((x<=0.) & (y<=0.)) + (1)*((x>0.) & (y<=0.)) + 161.45*((x>0.) & (y>0.)) + 1*((x<=0.) & (y>0.))
#coefficient

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

# ╔═╡ 1b8a3096-64c4-11eb-21c6-6b0b43f7e84d
e1 = triMesh(dofs=N);

# ╔═╡ 1480af62-64ad-11eb-28a7-d34225155b25
generate_tensor_mesh(e1);

# ╔═╡ ddd54870-673a-11eb-01a5-ffe24cc7a2ad
A,L,rhs = global_assemble(e1);

# ╔═╡ fc555ece-6970-11eb-01ea-c99033916c7d
λ_L,=eigs(L,nev=N,which=:SM);

# ╔═╡ ef26302c-68a4-11eb-29d7-730bda510b12
λ_A,=eigs(A,nev=N,which=:SM);

# ╔═╡ 89f1d0aa-6723-11eb-1b3b-e5a761f2fa18
u,ch = solve_fem(e1,A,L,rhs,"cg");

# ╔═╡ 6ae5e60a-64f9-11eb-242a-c9e6bf68f24c
begin
	pl = e1.pointlist; tl = e1.trianglelist;
	plotpair(pl,tl,u)
end

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

# ╔═╡ ae8804d4-6971-11eb-0cbf-cfc0b717b388
begin
	λ_AL,v_AL=eigen(AL);
	λ_AL=sort(real.(λ_AL));
	v_AL=real.(v_AL);
	v_AL = v_AL[:, sortperm(λ_AL)];
	"Eigenvalues and eigenvectors of AL calculated"
end

# ╔═╡ 7cff19fe-696f-11eb-27e0-eb6fb8451d61
plotspectra([λ_A,λ_L,λ_AL],["A","L","A_L"])

# ╔═╡ Cell order:
# ╟─d515e2f4-640a-11eb-1a73-159e6d21533e
# ╠═aa0f9f7e-6409-11eb-3125-1301a8b20ef9
# ╟─3a4fc858-647c-11eb-0a23-e7c8ec8348c6
# ╟─2c2f9998-647b-11eb-3ddc-11f90d029e3c
# ╟─12755762-688c-11eb-14a3-479950c1a3a1
# ╠═ed36e1ac-696f-11eb-00b8-5d6cc89a731e
# ╟─8852442c-6481-11eb-281c-e53ac9aa06f3
# ╠═cd77adb2-64bd-11eb-1700-8dc2f703553d
# ╟─85461a64-71b6-11eb-1153-01c5012c2d3d
# ╟─ec9dea62-71b7-11eb-0c8d-ddbeaad47e00
# ╟─8eae2aa4-71b8-11eb-1e08-eb530bbf3b44
# ╟─9fc55114-71b8-11eb-256e-0fc33ea16a4d
# ╟─829c6982-71b9-11eb-3e41-77e1d8d323f4
# ╟─99b9c812-71bb-11eb-0511-972a2663b3a4
# ╟─3a438aa0-71bc-11eb-1eac-df30d59b8104
# ╟─797a74d2-71bb-11eb-0766-23e66d9a5c18
# ╟─8cf8b7da-71bb-11eb-12ae-cb2d8054948f
# ╟─92281bfa-71bc-11eb-2dee-1907ead77773
# ╟─6ea24578-71be-11eb-3a8d-33fe0014b0e8
# ╟─71099b0e-71be-11eb-095f-2581ed5e01c3
# ╟─7bed953e-71be-11eb-0071-757715aab843
# ╟─87d83dae-71be-11eb-1c98-e729954a844f
# ╟─db509706-71be-11eb-35e4-d5823c530c2a
# ╟─18f2455c-71bf-11eb-3360-81ef505644e4
# ╟─cb8b1fdc-71c1-11eb-1c5a-79d4213317f0
# ╟─7aed1ca2-71c0-11eb-34e9-9b8c3f1d3445
# ╟─4520e0e2-71c3-11eb-19b0-9763903df14c
# ╟─abae667e-71c3-11eb-368b-cfd9e5295b7c
# ╟─194ae372-71c4-11eb-16a6-0dc3f89736e7
# ╟─96d1f4f2-71c4-11eb-140d-1382f39fe066
# ╟─a1f53f76-71c4-11eb-112f-0f9c00f74390
# ╟─76a94e90-71c5-11eb-352b-d1cc8edee31e
# ╟─bfefdf7e-71c5-11eb-0d02-89d589d49170
# ╟─27c70514-71c6-11eb-1703-2d77910f2002
# ╟─0a40431a-71c7-11eb-2807-bbaec4e2c21c
# ╟─5c60b184-71c7-11eb-1673-37851c277024
# ╟─ea4e69f6-71cb-11eb-379f-23f54f16a48f
# ╟─fdc1ed30-71c8-11eb-3f5c-3554de976606
# ╠═6ae5e60a-64f9-11eb-242a-c9e6bf68f24c
# ╟─e66e4a5a-71c8-11eb-20d1-896fda01b251
# ╟─509b94ec-71ca-11eb-352c-d784033c5dd4
# ╠═c09b2a4a-6896-11eb-0289-cb57389c9a32
# ╠═21fcaac2-692d-11eb-38ca-0be1fae1718e
# ╟─e5cd2b6e-71d6-11eb-09e9-e38626e7efb7
# ╟─305e5024-71cb-11eb-12b5-31da272983dc
# ╟─4065df58-71cd-11eb-1250-d5b2c826e3cb
# ╟─9b351cb4-71cb-11eb-379a-eddd876a0c87
# ╠═7cff19fe-696f-11eb-27e0-eb6fb8451d61
# ╠═fc555ece-6970-11eb-01ea-c99033916c7d
# ╠═ef26302c-68a4-11eb-29d7-730bda510b12
# ╟─5b465b46-71cc-11eb-397d-85c3fcce10ac
# ╟─030777b6-71cf-11eb-2dce-1f5c5f99b597
# ╟─1452ebd4-71cf-11eb-3903-ef7234203ff4
# ╟─24c08780-71cd-11eb-2f2f-e1bc01955371
# ╟─1d240d4c-71cf-11eb-09b8-6983615a4992
# ╟─c9201808-71ce-11eb-0d1e-af3bdb657f89
# ╟─340f0494-71cf-11eb-0678-174d79dbef8c
# ╟─42273880-71cf-11eb-25e8-77e494cc5a45
# ╟─6c91897e-71d0-11eb-1530-8b1fe21692e6
# ╟─b3bc1022-71d1-11eb-22f1-e7ff814684a0
# ╟─d9221816-71d1-11eb-1cf4-9166a9c456fd
# ╟─bfa8ab34-71d1-11eb-0689-e1ef4f1552c2
# ╟─e462e1ce-71d1-11eb-1747-9374db5a9384
# ╟─f3dc1c24-71d1-11eb-0d0d-4d9a3e663f1c
# ╟─4f5e8b18-71d2-11eb-06bd-c1e048987ee7
# ╠═2e373cea-71f6-11eb-3930-09b0f1ea2c43
# ╟─b227b602-71d2-11eb-3166-cb56652dfa5f
# ╟─fff70918-71d3-11eb-1c48-93ada3377cb6
# ╟─3e01ad42-71d4-11eb-1f92-8fbc59171e97
# ╟─9d34f122-71d4-11eb-22aa-05c099444e41
# ╟─ad40ceb8-71d4-11eb-1f65-fb1d231752b2
# ╟─764e8752-71d5-11eb-3cc0-85899265e2df
# ╟─99153286-71d5-11eb-114d-953047d72616
# ╟─edad8744-71d5-11eb-3ff2-e108636b319d
# ╟─f5ff937e-71d5-11eb-3616-3b1e6ef88525
# ╟─a194d4fe-71ba-11eb-273b-0fcefe96bcc0
# ╟─aebfc210-71c9-11eb-08b5-f7aba9ac895f
# ╟─b56b808e-71d5-11eb-2f54-7f7c6eea5bbc
# ╟─383a3b3e-66e8-11eb-1594-5934cb8587c5
# ╠═1480af62-64ad-11eb-28a7-d34225155b25
# ╟─79af845a-64d8-11eb-3458-1f654490b456
# ╟─f906340a-64f5-11eb-2dd3-7110e723d06c
# ╟─8a1f0418-64f7-11eb-00bc-27f5c923357c
# ╟─48c11296-64cc-11eb-12e6-c95142efa7c1
# ╟─3309239c-6723-11eb-01ac-ab241fb96a42
# ╠═1b8a3096-64c4-11eb-21c6-6b0b43f7e84d
# ╠═ddd54870-673a-11eb-01a5-ffe24cc7a2ad
# ╠═89f1d0aa-6723-11eb-1b3b-e5a761f2fa18
# ╠═abff15fe-6890-11eb-27ca-258853f71f8c
# ╠═bf263a14-6957-11eb-1746-1741c11adf4d
# ╠═a4bbc224-6949-11eb-3e01-89f56bbf93a4
# ╠═1ef89e6c-6969-11eb-22ca-bf8042ac79e0
# ╠═0059f2e4-692d-11eb-2f18-1987e20dc34b
# ╟─ae8804d4-6971-11eb-0cbf-cfc0b717b388
