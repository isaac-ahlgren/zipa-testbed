/* Reed Solomon Coding for glyphs
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
 *
 * Source code is available at http://rscode.sourceforge.net
 *
 * Commercial licensing is available under a separate license, please
 * contact author for details.
 *
 */

/****************************************************************
  
  Below is NPAR, the only compile-time parameter you should have to
  modify.
  
  It is the number of parity bytes which will be appended to
  your data to create a codeword.

  Note that the maximum codeword size is 255, so the
  sum of your message length plus parity should be less than
  or equal to this maximum limit.

  In practice, you will get slooow error correction and decoding
  if you use more than a reasonably small number of parity bytes.
  (say, 10 or 20)

  ****************************************************************/
/****************************************************************/




#define TRUE 1
#define FALSE 0

typedef unsigned long BIT32;
typedef unsigned short BIT16;

struct ReedSolomon_Instance {
  unsigned int npar;    // Number of parity symbols
  unsigned int max_deg; // Maximum degree of various polynomials
  int* pBytes;          // Encoder parity bytes
  int* synBytes;        // Syndrome bytes
  int* genPoly;         // The generator polynomial

  /* error locations found using Chien's search*/
  int ErrorLocs[256];
  int NErrors;

  /* erasure flags */
  int ErasureLocs[256];
  int NErasures;

  /* The Error Locator Polynomial, also known as Lambda or Sigma. Lambda[0] == 1 */
  int* Lambda;

  /* The Error Evaluator Polynomial */
  int* Omega;
};

/* **************************************************************** */

/* print debugging info */
extern int DEBUG;

/* flag for if the tables for the galois field are already computed */
extern int INITIALIZED;

/* Reed Solomon encode/decode routines */
void initialize_ecc (void);
struct ReedSolomon_Instance* initialize_rs_instance(int npar);
void free_rs_instance(struct ReedSolomon_Instance* rs);
int check_syndrome (struct ReedSolomon_Instance* rs);
void decode_data (unsigned char data[], int nbytes, struct ReedSolomon_Instance* rs);
void encode_data (unsigned char msg[], int nbytes, unsigned char dst[], struct ReedSolomon_Instance* rs);

/* CRC-CCITT checksum generator */
BIT16 crc_ccitt(unsigned char *msg, int len);

/* galois arithmetic tables */
extern int gexp[];
extern int glog[];

void init_galois_tables (void);
int ginv(int elt); 
int gmult(int a, int b);


/* Error location routines */
int correct_errors_erasures (unsigned char codeword[], int csize,int nerasures, int erasures[], struct ReedSolomon_Instance* rs);

/* polynomial arithmetic */
void add_polys(int dst[], int src[], int max_deg);
void scale_poly(int k, int poly[], int max_deg);
void mult_polys(int dst[], int p1[], int p2[], int max_deg);

void copy_poly(int dst[], int src[], int max_deg);
void zero_poly(int poly[], int max_deg);
