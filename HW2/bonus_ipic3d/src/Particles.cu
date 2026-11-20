#include "Particles.h"
#include "Alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



// ===========================================================================================
// HW2 bonus: GPU particle mover
// ===========================================================================================

#define CUDA_CHECK(stmt)                                                           \
    do {                                                                           \
        cudaError_t err_ = (stmt);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt, __FILE__,       \
                    __LINE__, cudaGetErrorString(err_));                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

/** Grid quantities the kernel needs, with the same types as in struct grid, so that every
    expression of the CPU mover is evaluated with the same precision on the GPU. */
struct MoverGrid {
    int nyn, nzn;
    double xStart, yStart, zStart;
    double Lx, Ly, Lz;
    FPfield invdx, invdy, invdz, invVOL;
    bool periodicX, periodicY, periodicZ;
};

/** The particles do not interact in the mover and the fields are constant during a call,
    so one thread can advance one particle through all of its sub-cycles. */
__global__ void mover_kernel(FPpart* px, FPpart* py, FPpart* pz,
                             FPpart* pu, FPpart* pv, FPpart* pw, long nop,
                             const FPfield* Ex, const FPfield* Ey, const FPfield* Ez,
                             const FPfield* Bx, const FPfield* By, const FPfield* Bz,
                             const FPfield* XN, const FPfield* YN, const FPfield* ZN,
                             MoverGrid g, int n_sub_cycles, int NiterMover,
                             FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2)
{
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nop) return;

    // particle state lives in registers during the whole call
    FPpart x = px[i], y = py[i], z = pz[i];
    FPpart u = pu[i], v = pv[i], w = pw[i];

    FPpart omdtsq, denom, ut, vt, wt, udotb;
    FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    FPpart xptilde, yptilde, zptilde, uptilde = 0, vptilde = 0, wptilde = 0;

    // flat index of node (a, b, c); arrays are stored row-major with sizes (nxn, nyn, nzn)
    #define IDX(a, b, c) (((a) * g.nyn + (b)) * g.nzn + (c))

    for (int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {
        xptilde = x;
        yptilde = y;
        zptilde = z;
        // calculate the average velocity iteratively
        for (int innter = 0; innter < NiterMover; innter++) {
            // interpolation G-->P
            const int ix = 2 + int((x - g.xStart) * g.invdx);
            const int iy = 2 + int((y - g.yStart) * g.invdy);
            const int iz = 2 + int((z - g.zStart) * g.invdz);

            // calculate weights
            xi[0]   = x - XN[IDX(ix - 1, iy, iz)];
            eta[0]  = y - YN[IDX(ix, iy - 1, iz)];
            zeta[0] = z - ZN[IDX(ix, iy, iz - 1)];
            xi[1]   = XN[IDX(ix, iy, iz)] - x;
            eta[1]  = YN[IDX(ix, iy, iz)] - y;
            zeta[1] = ZN[IDX(ix, iy, iz)] - z;
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * g.invVOL;

            Exl = 0.0; Eyl = 0.0; Ezl = 0.0; Bxl = 0.0; Byl = 0.0; Bzl = 0.0;
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        const int n = IDX(ix - ii, iy - jj, iz - kk);
                        Exl += weight[ii][jj][kk] * Ex[n];
                        Eyl += weight[ii][jj][kk] * Ey[n];
                        Ezl += weight[ii][jj][kk] * Ez[n];
                        Bxl += weight[ii][jj][kk] * Bx[n];
                        Byl += weight[ii][jj][kk] * By[n];
                        Bzl += weight[ii][jj][kk] * Bz[n];
                    }

            // end interpolation
            omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
            denom = 1.0 / (1.0 + omdtsq);
            // solve the position equation
            ut = u + qomdt2 * Exl;
            vt = v + qomdt2 * Eyl;
            wt = w + qomdt2 * Ezl;
            udotb = ut * Bxl + vt * Byl + wt * Bzl;
            // solve the velocity equation
            uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
            // update position
            x = xptilde + uptilde * dto2;
            y = yptilde + vptilde * dto2;
            z = zptilde + wptilde * dto2;
        } // end of iteration
        // update the final position and velocity
        u = 2.0 * uptilde - u;
        v = 2.0 * vptilde - v;
        w = 2.0 * wptilde - w;
        x = xptilde + uptilde * dt_sub_cycling;
        y = yptilde + vptilde * dt_sub_cycling;
        z = zptilde + wptilde * dt_sub_cycling;

        // X-DIRECTION: BC particles
        if (x > g.Lx) {
            if (g.periodicX) { x = x - g.Lx; }
            else { u = -u; x = 2 * g.Lx - x; }
        }
        if (x < 0) {
            if (g.periodicX) { x = x + g.Lx; }
            else { u = -u; x = -x; }
        }
        // Y-DIRECTION: BC particles
        if (y > g.Ly) {
            if (g.periodicY) { y = y - g.Ly; }
            else { v = -v; y = 2 * g.Ly - y; }
        }
        if (y < 0) {
            if (g.periodicY) { y = y + g.Ly; }
            else { v = -v; y = -y; }
        }
        // Z-DIRECTION: BC particles
        if (z > g.Lz) {
            if (g.periodicZ) { z = z - g.Lz; }
            else { w = -w; z = 2 * g.Lz - z; }
        }
        if (z < 0) {
            if (g.periodicZ) { z = z + g.Lz; }
            else { w = -w; z = -z; }
        }
    } // end of subcycling
    #undef IDX

    px[i] = x; py[i] = y; pz[i] = z;
    pu[i] = u; pv[i] = v; pw[i] = w;
}

/** Initialise the CUDA context, so that its start-up cost is not counted in the mover time. */
void gpu_warmup()
{
    CUDA_CHECK(cudaFree(0));
}

/** particle mover on the GPU. For simplicity all device memory is allocated, filled and freed
    inside this function: the particles and the fields are copied to the GPU, the kernel runs,
    and the particles are copied back, so interpP2G() and the rest of the code stay on the CPU. */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    std::cout << "***  GPU MOVER with SUBCYCLYING "<< part->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // same scalar set-up as in mover_PC()
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    MoverGrid g;
    g.nyn = grd->nyn;  g.nzn = grd->nzn;
    g.xStart = grd->xStart; g.yStart = grd->yStart; g.zStart = grd->zStart;
    g.Lx = grd->Lx; g.Ly = grd->Ly; g.Lz = grd->Lz;
    g.invdx = grd->invdx; g.invdy = grd->invdy; g.invdz = grd->invdz; g.invVOL = grd->invVOL;
    g.periodicX = param->PERIODICX; g.periodicY = param->PERIODICY; g.periodicZ = param->PERIODICZ;

    const long nop = part->nop;
    const size_t pBytes = nop * sizeof(FPpart);
    const size_t fBytes = (size_t) grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);

    cudaEvent_t tStart, tCopied, tKernel, tEnd;
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tCopied));
    CUDA_CHECK(cudaEventCreate(&tKernel));
    CUDA_CHECK(cudaEventCreate(&tEnd));
    CUDA_CHECK(cudaEventRecord(tStart));

    // device memory: 6 particle arrays + 6 field arrays + 3 node-coordinate arrays
    FPpart *dx, *dy, *dz, *du, *dv, *dw;
    FPfield *dEx, *dEy, *dEz, *dBx, *dBy, *dBz, *dXN, *dYN, *dZN;
    CUDA_CHECK(cudaMalloc(&dx, pBytes)); CUDA_CHECK(cudaMalloc(&dy, pBytes)); CUDA_CHECK(cudaMalloc(&dz, pBytes));
    CUDA_CHECK(cudaMalloc(&du, pBytes)); CUDA_CHECK(cudaMalloc(&dv, pBytes)); CUDA_CHECK(cudaMalloc(&dw, pBytes));
    CUDA_CHECK(cudaMalloc(&dEx, fBytes)); CUDA_CHECK(cudaMalloc(&dEy, fBytes)); CUDA_CHECK(cudaMalloc(&dEz, fBytes));
    CUDA_CHECK(cudaMalloc(&dBx, fBytes)); CUDA_CHECK(cudaMalloc(&dBy, fBytes)); CUDA_CHECK(cudaMalloc(&dBz, fBytes));
    CUDA_CHECK(cudaMalloc(&dXN, fBytes)); CUDA_CHECK(cudaMalloc(&dYN, fBytes)); CUDA_CHECK(cudaMalloc(&dZN, fBytes));

    // host -> device
    CUDA_CHECK(cudaMemcpy(dx, part->x, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, part->y, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, part->z, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(du, part->u, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, part->v, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, part->w, pBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEx, field->Ex_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEy, field->Ey_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEz, field->Ez_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBx, field->Bxn_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBy, field->Byn_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBz, field->Bzn_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXN, grd->XN_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dYN, grd->YN_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dZN, grd->ZN_flat, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(tCopied));

    // one thread per particle
    const int threads = 256;
    const int blocks = (int) ((nop + threads - 1) / threads);
    mover_kernel<<<blocks, threads>>>(dx, dy, dz, du, dv, dw, nop,
                                      dEx, dEy, dEz, dBx, dBy, dBz, dXN, dYN, dZN,
                                      g, part->n_sub_cycles, part->NiterMover,
                                      dt_sub_cycling, dto2, qomdt2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(tKernel));

    // device -> host
    CUDA_CHECK(cudaMemcpy(part->x, dx, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->y, dy, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->z, dz, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->u, du, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->v, dv, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->w, dw, pBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(tEnd));
    CUDA_CHECK(cudaEventSynchronize(tEnd));

    float msCopyIn, msKernel, msTotal;
    CUDA_CHECK(cudaEventElapsedTime(&msCopyIn, tStart, tCopied));
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, tCopied, tKernel));
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, tStart, tEnd));
    std::cout << "GPU mover species " << part->species_ID << ": alloc+H2D " << msCopyIn
              << " ms, kernel " << msKernel << " ms, total with D2H " << msTotal << " ms" << std::endl;

    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(du); cudaFree(dv); cudaFree(dw);
    cudaFree(dEx); cudaFree(dEy); cudaFree(dEz); cudaFree(dBx); cudaFree(dBy); cudaFree(dBz);
    cudaFree(dXN); cudaFree(dYN); cudaFree(dZN);
    cudaEventDestroy(tStart); cudaEventDestroy(tCopied); cudaEventDestroy(tKernel); cudaEventDestroy(tEnd);

    return(0); // exit succcesfully
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
