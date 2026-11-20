/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// HW2 bonus: write the final particle state of every species to <prefix>_s<is>.bin
// (long nop, then x,y,z,u,v,w as FPpart arrays) and print a per-species checksum, so that the
// CPU and the GPU mover can be compared with compare_particles.
static void dump_particles(const char* prefix, particles* part, int ns)
{
    for (int is = 0; is < ns; is++) {
        const long nop = part[is].nop;
        char name[1024];
        snprintf(name, sizeof(name), "%s_s%d.bin", prefix, is);
        FILE* f = fopen(name, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", name); return; }
        fwrite(&nop, sizeof(long), 1, f);
        FPpart* arrays[6] = { part[is].x, part[is].y, part[is].z, part[is].u, part[is].v, part[is].w };
        double sums[6] = {0, 0, 0, 0, 0, 0};
        for (int a = 0; a < 6; a++) {
            fwrite(arrays[a], sizeof(FPpart), nop, f);
            for (long i = 0; i < nop; i++) sums[a] += arrays[a][i];
        }
        fclose(f);
        std::cout << "CHECKSUM species " << is << " nop=" << nop << std::scientific
                  << " sum(x)=" << sums[0] << " sum(y)=" << sums[1] << " sum(z)=" << sums[2]
                  << " sum(u)=" << sums[3] << " sum(v)=" << sums[4] << " sum(w)=" << sums[5]
                  << std::defaultfloat << std::endl;
    }
}


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    // HW2 bonus: MOVER=gpu selects mover_PC_gpu(), anything else (default) the CPU mover_PC()
    const char* moverEnv = getenv("MOVER");
    const bool useGpu = moverEnv && strcmp(moverEnv, "gpu") == 0;
    std::cout << "Mover: " << (useGpu ? "GPU" : "CPU") << std::endl;
    if (useGpu) gpu_warmup();

    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++) {
            if (useGpu) mover_PC_gpu(&part[is],&field,&grd,&param);
            else        mover_PC(&part[is],&field,&grd,&param);
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    // HW2 bonus: dump the final particles if DUMP_PARTICLES=<prefix> is set
    const char* dumpPrefix = getenv("DUMP_PARTICLES");
    if (dumpPrefix) dump_particles(dumpPrefix, part, param.ns);

    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


