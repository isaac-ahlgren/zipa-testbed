/* 
 * Reed Solomon Encoder/Decoder 
 *
 * Copyright Henry Minsky (hqm@alum.mit.edu) 1991-2009
 *
 * This software library is licensed under terms of the GNU GENERAL
 * PUBLIC LICENSE
 *
 * RSCODE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RSCODE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Rscode.  If not, see <http://www.gnu.org/licenses/>.

 * Commercial licensing is available under a separate license, please
 * contact author for details.
 *
 * Source code is available at http://rscode.sourceforge.net
 */

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "ecc.h"

int DEBUG = FALSE;
int INITIALIZED = FALSE;

static void
compute_genpoly (int nbytes, struct ReedSolomon_Instance* rs);

/* Initialize lookup tables, polynomials, etc. */
void
initialize_ecc ()
{
    if (INITIALIZED == FALSE) {
        /* Initialize the galois field arithmetic tables */
        init_galois_tables();
        INITIALIZED = TRUE;
    }
}

/* FUNCTION CREATED FOR ZIPA TESTBED. ALL FURTHER MODIFICATIONS TO THE CODE WILL BE IN SERVICE OF MAKING THAT WORK */
struct ReedSolomon_Instance*
initialize_rs_instance(int npar) 
{
    /* Maximum degree of various polynomials. */
    int max_deg = 2*npar;

    // Initialize instance for npar
    struct ReedSolomon_Instance* rs = (struct ReedSolomon_Instance*) malloc(sizeof(struct ReedSolomon_Instance));
    rs->npar = npar;
    rs->max_deg = max_deg;

    rs->pBytes = (int*) malloc(sizeof(int)*max_deg);
    memset(rs->pBytes, 0, sizeof(int)*max_deg);

    rs->synBytes = (int*) malloc(sizeof(int)*max_deg);
    memset(rs->synBytes, 0, sizeof(int)*max_deg);

    rs->genPoly = (int*) malloc(2*sizeof(int)*max_deg);
    memset(rs->genPoly, 0, 2*sizeof(int)*max_deg);


    rs->NErasures = 0;
    rs->NErrors = 0;

    rs->Lambda = (int*) malloc(sizeof(int)*max_deg);
    memset(rs->Lambda, 0, sizeof(int)*max_deg);

    rs->Omega = (int*) malloc(sizeof(int)*max_deg);
    memset(rs->Omega, 0, sizeof(int)*max_deg);

    /* Compute the encoder generator polynomial */
    compute_genpoly(npar, rs);

    return rs;
}

/* FUNCTION CREATED FOR ZIPA TESTBED. ALL FURTHER MODIFICATIONS TO THE CODE WILL BE IN SERVICE OF MAKING THAT WORK */
void
free_rs_instance(struct ReedSolomon_Instance* rs) 
{
    free(rs->synBytes);
    free(rs->genPoly);
    free(rs->pBytes);
    free(rs->Lambda);
    free(rs->Omega);
    free(rs);
}

void
zero_fill_from (unsigned char buf[], int from, int to)
{
  int i;
  for (i = from; i < to; i++) buf[i] = 0;
}

/* debugging routines */
void
print_parity (struct ReedSolomon_Instance* rs)
{ 
  int i, npar = rs->npar;
  int* pBytes = rs->pBytes;
  printf("Parity Bytes: ");
  for (i = 0; i < npar; i++) 
    printf("[%d]:%x, ",i,pBytes[i]);
  printf("\n");
}


void
print_syndrome (struct ReedSolomon_Instance* rs)
{ 
  int i, npar = rs->npar;
  int* synBytes = rs->synBytes;
  printf("Syndrome Bytes: ");
  for (i = 0; i < npar; i++) 
    printf("[%d]:%x, ",i,synBytes[i]);
  printf("\n");
}

/* Append the parity bytes onto the end of the message */
void
build_codeword (unsigned char msg[], int nbytes, unsigned char dst[], struct ReedSolomon_Instance* rs)
{
  int i, npar = rs->npar;
	int* pBytes = rs->pBytes;

  for (i = 0; i < nbytes; i++) dst[i] = msg[i];
	
  for (i = 0; i < npar; i++) {
    dst[i+nbytes] = pBytes[npar-1-i];
  }
}
	
/**********************************************************
 * Reed Solomon Decoder 
 *
 * Computes the syndrome of a codeword. Puts the results
 * into the synBytes[] array.
 */
 
void
decode_data(unsigned char data[], int nbytes, struct ReedSolomon_Instance* rs)
{
  int i, j, sum, npar = rs->npar;
  int* synBytes = rs->synBytes;

  for (j = 0; j < npar;  j++) {
    sum	= 0;
    for (i = 0; i < nbytes; i++) {
      sum = data[i] ^ gmult(gexp[j+1], sum);
    }
    synBytes[j]  = sum;
  }
}


/* Check if the syndrome is zero */
int
check_syndrome (struct ReedSolomon_Instance* rs)
{
 int i, nz = 0, npar = rs->npar;
 int* synBytes = rs->synBytes;
 
 for (i =0 ; i < npar; i++) {
  if (synBytes[i] != 0) {
      nz = 1;
      break;
  }
 }
 return nz;
}


void
debug_check_syndrome (struct ReedSolomon_Instance* rs)
{	
  int i;
	int* synBytes = rs->synBytes;
  for (i = 0; i < 3; i++) {
    printf(" inv log S[%d]/S[%d] = %d\n", i, i+1, 
	   glog[gmult(synBytes[i], ginv(synBytes[i+1]))]);
  }
}


/* Create a generator polynomial for an n byte RS code. 
 * The coefficients are returned in the genPoly arg.
 * Make sure that the genPoly array which is passed in is 
 * at least n+1 bytes long.
 */

static void
compute_genpoly (int nbytes, struct ReedSolomon_Instance* rs)
{
  int i, tp[256], tp1[256], max_deg = rs->max_deg;
	int* genpoly = rs->genPoly;

  /* multiply (x + a^n) for n = 1 to nbytes */

  zero_poly(tp1, max_deg);
  tp1[0] = 1;

  for (i = 1; i <= nbytes; i++) {
    zero_poly(tp, max_deg);
    tp[0] = gexp[i];		/* set up x+a^n */
    tp[1] = 1;
	  
    mult_polys(genpoly, tp, tp1, max_deg);
    copy_poly(tp1, genpoly, max_deg);
  }
}

/* Simulate a LFSR with generator polynomial for n byte RS code. 
 * Pass in a pointer to the data array, and amount of data. 
 *
 * The parity bytes are deposited into pBytes[], and the whole message
 * and parity are copied to dest to make a codeword.
 * 
 */

void
encode_data (unsigned char msg[], int nbytes, unsigned char dst[], struct ReedSolomon_Instance* rs)
{
  int npar = rs->npar;
  int* genPoly = rs->genPoly;
  int* pBytes = rs->pBytes;
  int i, LFSR[npar+1],dbyte, j;
	
  for(i=0; i < npar+1; i++) LFSR[i]=0;

  for (i = 0; i < nbytes; i++) {
    dbyte = msg[i] ^ LFSR[npar-1];
    for (j = npar-1; j > 0; j--) {
      LFSR[j] = LFSR[j-1] ^ gmult(genPoly[j], dbyte);
    }
    LFSR[0] = gmult(genPoly[0], dbyte);
  }

  for (i = 0; i < npar; i++) 
    pBytes[i] = LFSR[i];
	
  build_codeword(msg, nbytes, dst, rs);
}

