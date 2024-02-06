/* Example use of Reed-Solomon library 
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
 * This same code demonstrates the use of the encodier and 
 * decoder/error-correction routines. 
 *
 * We are assuming we have at least four bytes of parity (NPAR >= 4).
 * 
 * This gives us the ability to correct up to two errors, or 
 * four erasures. 
 *
 * In general, with E errors, and K erasures, you will need
 * 2E + K bytes of parity to be able to correct the codeword
 * back to recover the original message data.
 *
 * You could say that each error 'consumes' two bytes of the parity,
 * whereas each erasure 'consumes' one byte.
 *
 * Thus, as demonstrated below, we can inject one error (location unknown)
 * and two erasures (with their locations specified) and the 
 * error-correction routine will be able to correct the codeword
 * back to the original message.
 * */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ecc.h"
 
unsigned char msg[256];
unsigned char codeword[256];
 
/* Some debugging routines to introduce errors or erasures
   into a codeword. 
   */

/* Testing random message for error correction capabilities */
void make_random_key(char* msg, int byte_length) {
  for (int i = 0; i < byte_length; i++)
  {
    msg[i] = (char) (rand() % 256);
  }
  msg[byte_length] = '\0';
  //printf("%s\n", msg);
}

/* Introduce a byte error at LOC */
void
byte_err (int err, int loc, unsigned char *dst)
{
  printf("Adding Error at loc %d, data %#x\n", loc, dst[loc-1]);
  dst[loc-1] ^= err;
}

/* Pass in location of error (first byte position is
   labeled starting at 1, not 0), and the codeword.
*/
void
byte_erasure (int loc, unsigned char dst[], int cwsize, int erasures[]) 
{
  printf("Erasure at loc %d, data %#x\n", loc, dst[loc-1]);
  dst[loc-1] = 0;
}


#define ML (sizeof (msg) + NPAR)

int
main (int argc, char *argv[])
{
  srand(666);
  int npar = 4;
  int length_of_key = 8;
  int iterations = 500;
  int erasures[16];
  int nerasures = 0;

  initialize_ecc ();

  struct ReedSolomon_Instance* rs = initialize_rs_instance(npar);

  int number_of_successes = 0;
  for (int i = 0; i < iterations; i++)
  {
      make_random_key(msg, length_of_key);
 
      /* ************** */
 
      /* Encode data into codeword, adding NPAR parity bytes */
      encode_data(msg, length_of_key+1, codeword, rs);
 
      printf("Encoded data is: \"%s\"\n", codeword);

      codeword[0] = 0;
      codeword[1] = 0;
      //codeword[2] = 0;
      //codeword[3] = 0;
      /* Now decode -- encoded codeword size must be passed */
      decode_data(codeword, length_of_key + 1 + npar, rs);
      

      /* check if syndrome is all zeros */
      if (check_syndrome (rs) != 0) {
        correct_errors_erasures (codeword, 
			     length_of_key + 1 + npar,
			     nerasures, 
			     erasures,
           rs);
        printf("Corrected codeword: \"%X\"\n", codeword);
      }

      if (strcmp(codeword, msg) == 0) {
        number_of_successes++;
      }
  }
  printf("%d\n", number_of_successes / iterations);
  free_rs_instance(rs);
  exit(0);
}

