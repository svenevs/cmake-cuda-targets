///////////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, IT IS THE PROPERTY OF NVIDIA.  Comes from:         //
// https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-trianglescounting-example //
///////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>

#include <nvgraph.h>

#define check( a ) \
{\
    nvgraphStatus_t status = (a);\
    if ( (status) != NVGRAPH_STATUS_SUCCESS) {\
        printf("ERROR : %d in %s : %d\n", status, __FILE__ , __LINE__ );\
        exit(0);\
    }\
}

int main(int argc, char **argv)
{
    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;

    // Init host data
    CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

    // Undirected graph:
    // 0       2-------4
    //  \     / \     / \
    //   \   /   \   /   \
    //    \ /     \ /     \
    //     1-------3-------5
    // 3 triangles
    // CSR of lower triangular of adjacency matrix:
    const size_t n = 6, nnz = 8;
    int source_offsets[] = {0, 0, 1, 2, 4, 6, 8};
    int destination_indices[] = {0, 1, 1, 2, 2, 3, 3, 4};

    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets;
    CSR_input->destination_indices = destination_indices;
    // Set graph connectivity
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));

    uint64_t trcount = 0;
    check(nvgraphTriangleCount(handle, graph, &trcount));
    printf("Triangles count: %" PRIu64 "\n", trcount);

    free(CSR_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    return 0;
}
