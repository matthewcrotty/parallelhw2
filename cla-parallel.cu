/*********************************************************************/
//
// 02/01/2022: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
int gi[bits] = {0};
int pi[bits] = {0};
int ci[bits] = {0};

int sumi[bits] = {0};

int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;


void read_input()
{
  char* in1 = (char *)calloc(input_size+1, sizeof(char));
  char* in2 = (char *)calloc(input_size+1, sizeof(char));

  if( 1 != scanf("%s", in1))
    {
      printf("Failed to read input 1\n");
      exit(-1);
    }
  if( 1 != scanf("%s", in2))
    {
      printf("Failed to read input 2\n");
      exit(-1);
    }

  hex1 = grab_slice_char(in1,0,input_size+1);
  hex2 = grab_slice_char(in2,0,input_size+1);

  free(in1);
  free(in2);
}

// Unrolled loop, each thread gets 1 bit.
__global__
void compute_gp_c(int* gi_c, int* pi_c, int* bin1_c, int* bin2_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < bits){
        gi_c[index] = bin1_c[index] & bin2_c[index];
        pi_c[index] = bin1_c[index] | bin2_c[index];
    }
}

// Unrolled loop, each thread gets 1 bit of ngroups, which reads a block_size
// chunk from gi and pi.
__global__
void compute_group_gp_c(int* ggj_c, int* gpj_c, int* gi_c, int* pi_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ngroups){
        int jstart = index * block_size;
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = gi_c[jstart + i];
            for(int ii = block_size-1; ii > i; ii--){
                mult &= pi_c[jstart + ii];
            }
            sum |= mult;
        }
        ggj_c[index] = sum;

        int mult = pi_c[jstart];
        for(int i = 1; i < block_size; i++){
            mult &= pi_c[jstart + i];
        }
        gpj_c[index] = mult;
    }

}

// Unrolled loop, each thread gets 1 bit of nsections, which reads a block_size
// chunk from ggj and gpj
__global__
void compute_section_gp_c(int* sgk_c, int* spk_c, int* ggj_c, int* gpj_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nsections){
        int kstart = index * block_size;
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = ggj_c[kstart + i];
            for(int ii = block_size-1; ii > i; ii--){
                mult &= gpj_c[kstart + ii];
            }
            sum |= mult;
        }
        sgk_c[index] = sum;

        int mult = gpj_c[kstart];
        for(int i = 1; i < block_size; i++){
            mult &= gpj_c[kstart + 1];
        }
        spk_c[index] = mult;
    }
}

// Unrolled loop, each thread gets 1 bit of nsupersections, which reads a block_size
// chunk from sgk and spk
__global__
void compute_super_section_gp_c(int* ssgl_c, int* sspl_c, int* sgk_c, int* spk_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nsupersections){
        int lstart = index * block_size;
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = sgk_c[lstart + i];
            for(int ii = block_size-1; ii > i; ii--){
                mult &= spk_c[lstart + ii];
            }
            sum |= mult;
        }
        ssgl_c[index] = sum;

        int mult = spk_c[lstart];
        for(int i = 1; i < block_size; i++){
            mult &= spk_c[lstart + 1];
        }
        sspl_c[index] = mult;
    }
}

// Unrolled loop, each thread gets 1 bit of nsupersupersections, which reads a block_size
// chunk from ssgl and sspl
__global__
void compute_super_super_section_gp_c(int* sssgm_c, int* ssspm_c, int* ssgl_c, int* sspl_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nsupersupersections){
        int mstart = index * block_size;
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = ssgl_c[mstart + i];
            for(int ii = block_size-1; ii > i; ii--){
                mult &= sspl_c[mstart + ii];
            }
            sum |= mult;
        }
        sssgm_c[index] = sum;

        int mult = sspl_c[mstart];
        for(int i = 1; i < block_size; i++){
            mult &= sspl_c[mstart + 1];
        }
        ssspm_c[index] = mult;
    }
}

// I dont think this one can be parelellized.
void compute_super_super_section_carry_c(int* ssscm_c, int* sssgm_c, int* ssspm_c){
    for(int m = 0; m < nsupersupersections; m++){
        int ssscmlast = 0;
        if(m == 0){
            ssscmlast = 0;
        } else{
            ssscmlast = ssscm_c[m-1];
        }
        ssscm_c[m] = sssgm_c[m] | (ssspm_c[m] & ssscmlast);
    }

}

// Each thread reads from ssscm and then iterates through 1 block
__global__
void compute_super_section_carry_c(int* sscl_c, int* ssgl_c, int* sspl_c, int* ssscm_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index == 0){
        sscl_c[0] = ssgl_c[0] | 0;
    }
    if(threadIdx.x < nsupersupersections){
        if(index != 0)
            sscl_c[index * block_size] = ssgl_c[index * block_size] | (sspl_c[index * block_size] & ssscm_c[index]);
        __syncthreads();
        index *= block_size;
        for(int l = 1; l < block_size; l++){
            if(index + l < nsupersections){
                sscl_c[index + l] = ssgl_c[index + l] | (sspl_c[index + l] & sscl_c[index + l -1]);
            }
        }
    }
}

// Each thread reads from sscl and iterates through 1 block
__global__
void compute_section_carry_c(int* sck_c, int* sgk_c, int* spk_c, int* sscl_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index == 0){
        sck_c[0] = sgk_c[0] | 0;
    }
    if(threadIdx.x < nsupersections){
        if(index != 0)
            sck_c[index * block_size] = sgk_c[index * block_size] | (spk_c[index * block_size] & sscl_c[index]);
        __syncthreads();
        index *= block_size;
        for(int k = 1; k < block_size; k++){
            if(index + k < nsections){
                sck_c[index + k] = sgk_c[index + k] | (spk_c[index + k] & sck_c[index + k - 1]);
            }
        }
    }
}

// Each thread reads from sck and iterates through 1 block
__global__
void compute_group_carry_c(int* gcj_c, int* ggj_c, int* gpj_c, int* sck_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index == 0){
        gcj_c[0] = ggc_c[0] | 0;
    }
    if(threadIdx.x < nsections){
        if(index != 0)
            gcj_c[index * block_size] = ggj_c[index * block_size] | (gpj_c[index * block_size] & sck_c[index]);
        __syncthreads();
        index *= block_size;
        for(int j = 1; j < block_size; j++){
            if(index + j < ngroups){
                gcj_c[index + j] = ggj_c[index + j] | (gpj_c[index + j] & gcj_c[index+j -1]);
            }
        }
    }
}

// Each thread reads from gcj and iterates through 1 block
__global__
void compute_carry_c(int* ci_c, int* gi_c, int* pi_c, int* gcj_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index == 0){
        ci_c[0] = gi_c[0] | 0;
    }
    if(threadIdx.x < ngroups){
        if(index != 0)
            ci_c[index * block_size] = gi_c[index * block_size] | (pi_c[index * block_size] & gcj_c[index-1]);
        __syncthreads();
        index *= block_size;
        for(int i = 1; i < block_size; i++){
            if(index + i < bits){
                ci_c[index + i] = gi_c[index + i] | (pi_c[index + i] & ci_c[index+i -1]);
            }
        }
    }
}

// Unrolled loop so each thread gets 1 bit to work with.
__global__
void compute_sum_c(int* sumi_c, int* bin1_c, int* bin2_c, int* ci_c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index != 0){
        if(index < bits)
            sumi_c[index] = bin1_c[index] ^ bin2_c[index] ^ ci_c[index - 1];
    }
}

void cla()
{

    // Malloc bin1 and bin2 so GPU can read, and copy data over.
    int* bin1_cuda, *bin2_cuda;
    cudaMallocManaged(&bin1_cuda, bits*sizeof(int));
    cudaMallocManaged(&bin2_cuda, bits*sizeof(int));

    for(int i = 0; i < bits; i++){
        bin1_cuda[i] = bin1[i];
        bin2_cuda[i] = bin2[i];
    }

    int blockSize = 256;

    // Call all kernels, allocating arrays as needed

    int* gi_cuda, *pi_cuda;
    cudaMallocManaged(&gi_cuda, bits*sizeof(int));
    cudaMallocManaged(&pi_cuda, bits*sizeof(int));

    compute_gp_c<<<(bits + blockSize -1)/blockSize, blockSize>>>(gi_cuda, pi_cuda, bin1_cuda, bin2_cuda);

    int* ggj_cuda, *gpj_cuda;
    cudaMallocManaged(&ggj_cuda, ngroups*sizeof(int));
    cudaMallocManaged(&gpj_cuda, ngroups*sizeof(int));

    compute_group_gp_c<<<(ngroups + blockSize -1)/blockSize, blockSize>>>(ggj_cuda, gpj_cuda, gi_cuda, pi_cuda);

    int* sgk_cuda, *spk_cuda;
    cudaMallocManaged(&sgk_cuda, nsections*sizeof(int));
    cudaMallocManaged(&spk_cuda, nsections*sizeof(int));

    compute_section_gp_c<<<(nsections + blockSize -1)/blockSize, blockSize>>>(sgk_cuda, spk_cuda, ggj_cuda, gpj_cuda);

    int* ssgl_cuda, *sspl_cuda;
    cudaMallocManaged(&ssgl_cuda, nsupersections*sizeof(int));
    cudaMallocManaged(&sspl_cuda, nsupersections*sizeof(int));

    compute_super_section_gp_c<<<(nsupersections + blockSize -1)/blockSize, blockSize>>>(ssgl_cuda, sspl_cuda, sgk_cuda, spk_cuda);

    int* sssgm_cuda, *ssspm_cuda;
    cudaMallocManaged(&sssgm_cuda, nsupersupersections*sizeof(int));
    cudaMallocManaged(&ssspm_cuda, nsupersupersections*sizeof(int));

    compute_super_super_section_gp_c<<<(nsupersupersections + blockSize -1)/blockSize, blockSize>>>(sssgm_cuda, ssspm_cuda, ssgl_cuda, sspl_cuda);

    int* ssscm_cuda;
    cudaMallocManaged(&ssscm_cuda, nsupersupersections*sizeof(int));
    compute_super_super_section_carry_c(ssscm_cuda, sssgm_cuda, ssspm_cuda);

    int* sscl_cuda;
    cudaMallocManaged(&sscl_cuda, nsupersections*sizeof(int));
    compute_super_section_carry_c<<<(nsupersupersections + blockSize -1)/blockSize, blockSize>>>(sscl_cuda, ssgl_cuda, sspl_cuda, ssscm_cuda);

    int* sck_cuda;
    cudaMallocManaged(&sck_cuda, nsections*sizeof(int));
    compute_section_carry_c<<<(nsupersections + blockSize -1)/blockSize, blockSize>>>(sck_cuda, sgk_cuda, spk_cuda, sscl_cuda);

    int* gcj_cuda;
    cudaMallocManaged(&gcj_cuda, ngroups*sizeof(int));
    compute_group_carry_c<<<(nsections + blockSize -1)/blockSize, blockSize>>>(gcj_cuda, ggj_cuda, gpj_cuda, sck_cuda);

    int* ci_cuda;
    cudaMallocManaged(&ci_cuda, bits*sizeof(int));
    compute_carry_c<<<(ngroups + blockSize -1)/blockSize, blockSize>>>(ci_cuda, gi_cuda, pi_cuda, gcj_cuda);

    int* sumi_cuda;
    cudaMallocManaged(&sumi_cuda, bits*sizeof(int));
    compute_sum_c<<<(bits + blockSize -1)/blockSize, blockSize>>>(sumi_cuda, bin1_cuda, bin2_cuda, ci_cuda);


    // Synchronize and copy result into sumi from sumi_c
    cudaDeviceSynchronize();

    sumi_cuda[0] = bin1_cuda[0] ^ bin2_cuda[0] ^ 0;

    for(int i = 0; i < bits; i++){
        sumi[i] = sumi_cuda[i];
    }

    // Free all cuda malloc'd memory.
    cudaFree(bin1_cuda);
    cudaFree(bin2_cuda);
    cudaFree(gi_cuda);
    cudaFree(pi_cuda);
    cudaFree(ggj_cuda);
    cudaFree(gpj_cuda);
    cudaFree(sgk_cuda);
    cudaFree(spk_cuda);
    cudaFree(ssgl_cuda);
    cudaFree(sspl_cuda);
    cudaFree(sssgm_cuda);
    cudaFree(ssspm_cuda);
    cudaFree(ssscm_cuda);
    cudaFree(sscl_cuda);
    cudaFree(sck_cuda);
    cudaFree(gcj_cuda);
    cudaFree(ci_cuda);
    cudaFree(sumi_cuda);
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
	{
	  printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
	  printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
	  return;
	}
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  unsigned long long start_time=clock_now(); // dummy clock reads to init
  unsigned long long end_time=clock_now();   // dummy clock reads to init

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
	     nsupersupersections, block_size );
      return(-1);
    }

  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }

  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }

  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';

  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  start_time = clock_now();
  ripple_carry_adder();
  end_time = clock_now();

  printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);

  // free bin conversion of hex inputs
  free(bin1);
  free(bin2);

  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }

  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }

  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose==1 )
    printf("%s\n",hexSum);

  free(hexSum);


  return 1;
}
