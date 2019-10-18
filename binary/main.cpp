#include <stdio.h>
#include <math.h>
// #include <boost/random/uniform_01.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <ctime>
#include <string>
#include <cstring>
//#include <iterator>
// #include <boost/filesystem>
#include "globals.h"

using namespace::std ;
std::string folderName;

void ProgressBar(float progress, float me, float mi) {
    int barWidth = 31;
    std::cout << "Progress: [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)  std::cout << "\u25a0"; //std::cout << "=";
      else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% done | mE = " << me << " mI = " << mi << "\r";
    std::cout.flush();
    if(progress == 1.) std::cout << std::endl;
}

void ProgressBar(float progress, float me, float mi, float m0) {
    int barWidth = 31;
    std::cout << "Progress: [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)  std::cout << "\u25a0"; //std::cout << "=";
      else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% done | m0 = " << m0 << " mE = " << me << " mI = " << mi << "\r";
    std::cout.flush();
    if(progress == 1.) std::cout << std::endl;
}


void M1ComponentI(vector<double> &x, unsigned int n, double* m1, double* phase) {
  double dPhi = M_PI / (double)n;
  double xCord = 0, yCord = 0;
  for(unsigned int i = NE; i < NE + n; i++) {
    xCord += x[i] * cos(2.0 * i * dPhi);
    yCord += x[i] * sin(2.0 * i * dPhi);
  }
  *m1 = (2.0 / (double)n) * sqrt(xCord * xCord + yCord * yCord);
  *phase = 0.5 * atan2(yCord, xCord);
  if(*phase < 0) {
    *phase = *phase + M_PI;
  }
}

void M1ComponentE(vector<double> &x, unsigned int n, double* m1, double* phase) {
  double dPhi = M_PI / (double)n;
  double xCord = 0, yCord = 0;
  for(unsigned int i = 0; i < n; i++) {
    xCord += x[i] * cos(2.0 * i * dPhi);
    yCord += x[i] * sin(2.0 * i * dPhi);
  }
  *m1 = (2.0 / (double)n) * sqrt(xCord * xCord + yCord * yCord);
  //  *m1 = (1.0 / (double)n) * sqrt(xCord * xCord + yCord * yCord);  
  *phase = 0.5 * atan2(yCord, xCord);
  if(*phase < 0) {
    *phase = *phase + M_PI;
  }
}

double UniformRand() {
    // printf("\n hello from space 0\n");
    return (double)rand() / (double)RAND_MAX ;
}

void Dummy() {
  printf("me too\n");
}

unsigned int RandFFNeuron() { //unsigned int min, unsigned int max) {
  unsigned int result = NFF;
  while(result == NFF) {
    result = (unsigned int) (rand() / (double) (RAND_MAX + 1UL) * (NFF + 1));
  }
  return result;
}

unsigned int RandENeuron() { //unsigned int min, unsigned int max) {
  unsigned int result = NE;
  while(result == NE) {
    result = (unsigned int) (rand() / (double) (RAND_MAX + 1UL) * (NE + 1));
  }
  return result;
}

unsigned int RandINeuron() { //unsigned int min, unsigned int max) {
  unsigned int result = NI;
  while(result == NI) {
    result = NE + (unsigned int) (rand() / (double) (RAND_MAX + 1UL) * (NI + 1));
  }
  return result;  
}


// double ConProb(double phiI, double phiJ, unsigned int N) {
//   double out = ((double)K / (double)N) * (1 + 2.0 * (recModulation / SQRT_K) * cos(2.0 * (phiI - phiJ)));
//   if((out < 0) | (out > 1)) {
//     cout << "connection probability not in range [0, 1]!" << endl;
//     exit(1);
//   }
//   return out;
// }


double ConProb(double phiI, double phiJ, unsigned int N, double recModulationAB) {
  double out = ((double)K / (double)N) * (1 + 2.0 * (recModulationAB / SQRT_K) * cos(2.0 * (phiI - phiJ)));
  if((out < 0) | (out > 1)) {
    cout << "connection probability not in range [0, 1]!" << endl;
    exit(1);
  }
  ////////////////////////////////////////////////////
   // out = ((double)K / (double)NE);
  ////////////////////////////////////////////////////
  return out;
}


double ConProbFF(double phiI, double phiJ, unsigned int N) {
  double out = ((double)K * cFF / (double)N) * (1 + 2.0 * (ffModulation / SQRT_KFF) * cos(2.0 * (phiI - phiJ)));
  if((out < 0) | (out > 1)) {
    cout << "connection probability not in range [0, 1]!" << endl;
    exit(1);
  }
  return out;
}


// generate FF PO ~ U(0, PI)

// void GenFFPOs() {
//   std::default_random_engine genFFDet(1234U);
//   std::uniform_real_distribution<double> UniformRand_det(0.0, 1.0);
//   double poFFi = M_PI * UniformRand_det(genFFDet);
//   for(uint i = 0; i < NFF; i++) {
//     poFF[i] = poFFi;
//   }
//   FILE *fpPOff;
//   fpPOff = fopen("poFF.dat", "wb");
//   unsigned int nElementsWritten;
//   nElementsWritten = fwrite(poFF, sizeof(double), NFF, fpPOff);
//   if (nElementsWritten != NFF) {
//       printf("NFF neq nElements\n");
//     }
//   fclose(fpPOff);
// }
  
double FFTuningCurve(unsigned int i, double phiOft) {
  // std::default_random_engine gen(1234U); // for reproducable FF PO
  // std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
  // poFFi = M_PI * UniformRand();
  // for(int i = 0; i < 10; i++) {
  //  double out = m0_ext + m1_ext * cos(2 * (phiOft - poFF[i]));
  double out = m0_ext + m1_ext * cos(2 * (phiOft - (double)i * M_PI / (double)NFF));
  if((out < 0) | (out > 1)) {
    cout << "external rate [0, 1]!" << endl;
    exit(1);
  }
  //  out = m0_ext;
  return out;
}

void GenRewiredSparseMat(unsigned int *conVec,  unsigned int rows, unsigned int clms, unsigned int* sparseVec, unsigned int* idxVec, unsigned int* nPostNeurons, unsigned int *IF_REWIRED_CON, unsigned int *IS_REWIRED_LINK) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      // printf("%llu %llu %llu %llu\n", i, j, i + clms * j, i + rows * j);
      if(conVec[i + rows * j]) { /* i --> j  */
        sparseVec[counter] = j;
	if(IF_REWIRED_CON[i + rows * j] == 1) {
	  IS_REWIRED_LINK[counter] = 1;
	}
        counter += 1;
        nPost += 1;
      }
    }
    nPostNeurons[i] = nPost;
  }
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }

 // printf("----------------------------------------\n");
 // printf("------------ ORIGINAL MATRIX in sparse -----------\n");
 // for(i = 0; i < rows; i++) {
 //   for(j = 0; j < clms; j++) {
 //     printf("%d ", (int)conVec[i + rows * j]);
 //   }
 //   printf("\n");
 // }
 // printf("----------------------------------------\n");
 // printf("----------------------------------------\n"); 
}


void GenSparseMat(unsigned int *conVec,  unsigned int rows, unsigned int clms, unsigned int* sparseVec, unsigned int* idxVec, unsigned int* nPostNeurons ) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      // printf("%llu %llu %llu %llu\n", i, j, i + clms * j, i + rows * j);
      if(conVec[i + rows * j]) { /* i --> j  */
        sparseVec[counter] = j;
        counter += 1;
        nPost += 1;
      }
    }
    nPostNeurons[i] = nPost;
  }
  
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }


 // printf("----------------------------------------\n");
 // printf("------------ ORIGINAL MATRIX in sparse -----------\n");
 // for(i = 0; i < rows; i++) {
 //   for(j = 0; j < clms; j++) {
 //     printf("%d ", (int)conVec[i + rows * j]);
 //   }
 //   printf("\n");
 // }
 // printf("----------------------------------------\n");
 // printf("----------------------------------------\n"); 
  
  
  // FILE *fp = NULL;
  // fp = fopen("check_prob.csv", "w");
  // unsigned int ccount = 0;
  // vector <unsigned int> shiftedVec(NE);
  // vector <unsigned int> shiftedVecBuff(NE);
  // for(i = 0; i < N_NEURONS; ++i) {
  //   ccount = 0;
  //   for(j =0; j < NE; ++j) {
  //     shiftedVecBuff[j] = conVec[j + i * N_NEURONS];
  //   }
  //   std::rotate(shiftedVecBuff.begin(), shiftedVecBuff.begin() + i, shiftedVecBuff.end());
  //   for(unsigned int l = 0; l < NE; l++) {
  //     shiftedVec[l] += shiftedVecBuff[l];
  //   }
  // }
  
  // FILE *fp2 = NULL;
  // fp2 = fopen("prob.csv", "w");
  // for(i = 0; i < NE; i++) {
  //   // printf("%llu:\n", i); //    %f\n", i, shiftedVec[i] / (double)NE);    
  //   //fprintf(fp, "%f\n", shiftedVec[i] / (double)NE);
  //   fprintf(fp2, "%f\n", ConProb(0.0, (double)i * M_PI / (double)NE, NE, recModulationEE));
  // }
  // fclose(fp2);
  // shiftedVecBuff.clear();
  // shiftedVec.clear();
}



void GenFFConMat() {
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
  unsigned long long int nConnections = 0;
  cout << "generating FF conmat" << endl;
  unsigned int *conMatFF = new unsigned int [(unsigned long int)NFF * N_NEURONS];

 for(unsigned long int i = 0; i < NFF; i++) {
   for(unsigned long int j = 0; j < N_NEURONS; j++) {
     conMatFF[i + NFF * j] = 0;
   }
 }

  for (unsigned long int i = 0; i < NFF; i++)  {
    for (unsigned long int j = 0; j < N_NEURONS; j++)  {
      // i --> j
      if(j < NE) { //E-to-E
	if(UniformRand(gen) <= ConProbFF(i * M_PI / (double)NFF, j * M_PI / (double)NE, NFF)) {
	  conMatFF[i + NFF * j] = 1;
  	  // conMatFF[i + NFF * j] = 1;
	  nConnections += 1;
	}
      }
      else {
	if(UniformRand(gen) <= (double)K * cFF / (double)NFF) {
	  conMatFF[i + NFF * j] = 1;
	  nConnections += 1;
	}
      }
    }
  }
  cout << "done" << endl;
  cout << "computing sparse rep" << endl;    
  sparseConVecFF = new unsigned int[nConnections];
  GenSparseMat(conMatFF, NFF, N_NEURONS, sparseConVecFF, idxVecFF, nPostNeuronsFF);
  cout << "done" << endl;


  
  FILE *ffp;
  ffp = fopen("kcount_ff.csv", "w");
  for(unsigned int lll = 0; lll < NFF; lll++) {
    fprintf(ffp, "%u\n", nPostNeuronsFF[lll]);
  }
  fclose(ffp);

 // printf("----------------------------------------\n");
 // printf("------------ ORIGINAL FF MATRIX -----------\n");
 // for(unsigned long int i = 0; i < NFF; i++) {
 //   for(unsigned long int j = 0; j < N_NEURONS; j++) {
 //     printf("%d ", (int)conMatFF[i + NFF * j]);
 //   }
 //   printf("\n");
 // }
 // printf("----------------------------------------\n");
 // printf("----------------------------------------\n"); 


  delete [] conMatFF;
 
  
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  unsigned long int nElementsWritten;
  printf("done\n#connections = %llu\n", nConnections);
  printf("writing to file ... "); fflush(stdout);
  fpSparseConVec = fopen("sparseConVecFF.dat", "wb");
  nElementsWritten = fwrite(sparseConVecFF, sizeof(*sparseConVecFF), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  if(nElementsWritten != nConnections) {
    printf("\n Error: All elements not written \n");
  }
  fpIdxVec = fopen("idxVecFF.dat", "wb");
  fwrite(idxVecFF, sizeof(*idxVecFF), NFF,  fpIdxVec);
  fclose(fpIdxVec);
  fpNpostNeurons = fopen("nPostNeuronsFF.dat", "wb");
  fwrite(nPostNeuronsFF, sizeof(*nPostNeuronsFF), NFF, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");



  //-----------------------------------------
  // printf("testing sparsevecff read\n");
  // unsigned long int dummy;
  // FILE *fpSparseConVecFF = fopen("sparseConVecFF.dat", "rb");
  // dummy = fread(sparseConVecFF, sizeof(*sparseConVecFF), nConnections, fpSparseConVecFF);
  //   printf("sparseConvec read: %lu %llu \n", dummy, nConnections);
  // if(dummy != nConnections) {
  //   printf("sparseConvec read error ? %lu %llu \n", dummy, nConnections);
  // fclose(fpSparseConVecFF);
  // }
}

void GenFixedFFConMat() {
  std::random_device rd;
  std::mt19937 g(rd());
  // std::default_random_engine gen(rd());
  // std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
  unsigned long long int nConnections = 0;
  cout << "generating fixed K FF conmat" << endl;
  unsigned int *conMatFF = new unsigned int [(unsigned long int)NFF * N_NEURONS];
  std::vector<unsigned int> nFFVector(NE);
  // std::vector<unsigned int> nEVector(NE);
  // std::vector<unsigned int> nIVector(NI);
  unsigned int KFF_E = (unsigned int)(cFF * K);
  unsigned int KFF_I = (unsigned int)(cFF * K);
  cout << "k_FF =" << KFF_E << endl;

  for(unsigned long int i = 0; i < NFF; i++) {
    for(unsigned long int j = 0; j < N_NEURONS; j++) {
      conMatFF[i + NFF * j] = 0;
    }
  }

  for(unsigned int j = 0; j < NFF; j++) {
    nFFVector[j] = j;
  }

  // for(unsigned int j = 0; j < NI; j++) {
  //   nIVector[j] = j + NE;
  // }

  // 0-to-E  
  for (unsigned long int i = 0; i < NE; i++)  {
    std::shuffle(nFFVector.begin(), nFFVector.end(), g);
    for(unsigned int k = 0; k < KFF_E; k++) {       // i --> k
      conMatFF[nFFVector[k] + NFF * i] = 1;
      nConnections += 1;
    }
  }
  // 0-to-I
  for (unsigned long int i = NE; i < N_NEURONS; i++)  {  
    std::shuffle(nFFVector.begin(), nFFVector.end(), g);    
    for(unsigned int k = 0; k < KFF_I; k++) {       // i --> k
      conMatFF[nFFVector[k] + NFF * i] = 1;      
      nConnections += 1;
    }
  }
  
  cout << "done" << endl;
  cout << "computing sparse rep" << endl;    
  sparseConVecFF = new unsigned int[nConnections];
  GenSparseMat(conMatFF, NFF, N_NEURONS, sparseConVecFF, idxVecFF, nPostNeuronsFF);
  cout << "done" << endl;

  FILE *ffp;
  ffp = fopen("kcount_ff.csv", "w");
  for(unsigned int lll = 0; lll < NFF; lll++) {
    fprintf(ffp, "%u\n", nPostNeuronsFF[lll]);
  }
  fclose(ffp);

 // printf("----------------------------------------\n");
 // printf("------------ ORIGINAL FF MATRIX -----------\n");
 // for(unsigned long int i = 0; i < NFF; i++) {
 //   for(unsigned long int j = 0; j < N_NEURONS; j++) {
 //     printf("%d ", (int)conMatFF[i + NFF * j]);
 //   }
 //   printf("\n");
 // }
 // printf("----------------------------------------\n");
 // printf("----------------------------------------\n"); 


  ffp = fopen("kff_indegree.csv", "w");
  std::vector<unsigned int> kffInDegree(N_NEURONS);// number of FF inputs to E and I
  // std::vector<unsigned int> kffInDegreeI(NI);  // number of FF inputs to I
  for(unsigned int lll = 0; lll < NFF; lll++) {
    for(unsigned int i = 0; i < N_NEURONS; i++) {
      kffInDegree[i] += conMatFF[lll + NFF * i];
    }
  }
  for(unsigned int lll = 0; lll < N_NEURONS; lll++) {
      fprintf(ffp, "%u\n", kffInDegree[lll]);
  }
  fclose(ffp);

  


  
  delete [] conMatFF;

 
  
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  unsigned long int nElementsWritten;
  printf("done\n#connections FF = %llu\n", nConnections);
  printf("writing to file ... "); fflush(stdout);
  fpSparseConVec = fopen("sparseConVecFF.dat", "wb");
  nElementsWritten = fwrite(sparseConVecFF, sizeof(*sparseConVecFF), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  if(nElementsWritten != nConnections) {
    printf("\n Error: All elements not written \n");
  }
  fpIdxVec = fopen("idxVecFF.dat", "wb");
  fwrite(idxVecFF, sizeof(*idxVecFF), NFF,  fpIdxVec);
  fclose(fpIdxVec);
  fpNpostNeurons = fopen("nPostNeuronsFF.dat", "wb");
  fwrite(nPostNeuronsFF, sizeof(*nPostNeuronsFF), NFF, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");
  
  
}

void AddConnections(unsigned int *conVec, double kappa) {
 unsigned long long int i, j, nConnections = 0, nElementsWritten;
 double addProb = 0; 
 std::random_device rd;
 std::default_random_engine gen(rd());
 std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
 vector<unsigned long> nPost(N_NEURONS);
 // READ PO's
 FILE *fpPOofNeurons = NULL, *ffp;
 fpPOofNeurons = fopen("poOfNeurons.dat", "rb");
 if(fpPOofNeurons == NULL) {
   printf("\n Error: file poOfNeurons.dat not found! \n");
   exit(1);
 }
 double *poOfNeurons = new double [NE];
 unsigned long int nPOsRead = 0; 
 nPOsRead = fread(poOfNeurons, sizeof(*poOfNeurons), NE, fpPOofNeurons);
 if(nPOsRead != NE) {
   printf("\n Error: All elements not written \n");
 }

 // for(int iii = 0; iii < 10; ++iii) { printf("%d ", poOfNeurons[iii]); }
 
 fclose(fpPOofNeurons);
 unsigned int *IF_REWIRED_CON = new unsigned int [(unsigned long int)N_NEURONS * N_NEURONS];
 for(i = 0; i < N_NEURONS; ++i) {
   for(j = 0; j < N_NEURONS; j++) {
     IF_REWIRED_CON[i + N_NEURONS * j] = 0;
   }
 }
 // ADDING NEW CONNECTIONS
    // double denom = NE - K - kappa * sqrt(K);
    // unsigned int wrngcntrEE = 0, wrngcntrIE = 0, wrngcntrEI = 0, wrngcntrII = 0;
    // for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    //   for (unsigned long int j = 0; j < N_NEURONS; j++)  {
    // 	if(conMat[i + N_NEURONS * j] != conVec[i + N_NEURONS * j]) {
    // 	  if(i < NE && j < NE)       { wrngcntrEE += 1;}
    // 	  else if(i < NE && j >= NE) { wrngcntrIE += 1;}
    // 	  else if(i >= NE && j < NE) { wrngcntrEI += 1;}
    // 	  else                       { wrngcntrII += 1;}	  	  
    // 	  //	  else (i >= NE && j >= NE) { wrngcntrII += 1;}	  
    // 	}
    //   }
    // }
    // printf("\n IN ADDCONFUNC BEFORE: \n oh la la! EE IE EI II= %u %u %u %u\n", wrngcntrEE, wrngcntrIE, wrngcntrEI, wrngcntrII);    
    double denom = NE - K - kappa * sqrt(K);
    for(i = 0; i < NE; i++) {
      for(j = 0; j < NE; j++) {
	// conVec[i + N_NEURONS * j]  = 0;
	//addProb = kappa * sqrt(K) * (NE - K - kappa * sqrt(K)) * (1 + cos(2.0 * (poOfNeurons[i] - poOfNeurons[j]))) / denom;
	// addProb = (double)K * (1 + kappa * cos(2.0 * (poOfNeurons[i] - poOfNeurons[j])) / sqrt((double)K) ) / (double)NE;

	if(conVec[i + NE * j] == 0) {
	  addProb = (kappa / rewiredEEWeight)* sqrt(K) * (1.0 + cos(2.0 * (poOfNeurons[i] - poOfNeurons[j]))) / denom;
	  if(addProb > 1 || addProb < 0) { printf("IN AddConnections() add prob not in range!!!\n"); exit(1); }
	  if(addProb >= UniformRand(gen)) {
	    conVec[i + N_NEURONS * j]  = 1;
	    IF_REWIRED_CON[i + N_NEURONS * j]  = 1;	    
	  }
	}
      }
    }

 // TESTING
    //  wrngcntrEE = 0; wrngcntrIE = 0; wrngcntrEI = 0; wrngcntrII = 0;
    // for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    //   for (unsigned long int j = 0; j < N_NEURONS; j++)  {
    // 	if(conMat[i + N_NEURONS * j] != conVec[i + N_NEURONS * j]) {
    // 	  if(i < NE && j < NE)       { wrngcntrEE += 1;}
    // 	  else if(i < NE && j >= NE) { wrngcntrIE += 1;}
    // 	  else if(i >= NE && j < NE) { wrngcntrEI += 1;}
    // 	  else                       { wrngcntrII += 1;}	  	  
    // 	  //	  else (i >= NE && j >= NE) { wrngcntrII += 1;}	  
    // 	}
    //   }
    // }
    // printf("\n IN ADDCONFUNC AFTER: \n oh la la! EE IE EI II= %u %u %u %u\n", wrngcntrEE, wrngcntrIE, wrngcntrEI, wrngcntrII);    
 
 // COUNT AFTER ADDING CONNECTIONS
 for(i = 0; i < N_NEURONS; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < N_NEURONS; j++) {
   for(i = 0; i < N_NEURONS; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
       nConnections += 1;
     }
   }
 }
 ffp = fopen("rewired_K_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp);
 
 // GENERATE SPARSE REPRESENTATION
 printf("#connections in rewired matrix = %llu\n", nConnections);
 cout << "computing sparse rep" << endl;
 sparseConVec = new unsigned int[nConnections];
 IS_REWIRED_LINK = new unsigned int[nConnections];
 GenRewiredSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons, IF_REWIRED_CON, IS_REWIRED_LINK);
 cout << "done" << endl;
 // WRITE TO FILE
 FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons, *fpIsRewiredLink;
 printf("done\n #connections = %llu\n", nConnections);
 printf("writing to file ... "); fflush(stdout);
 // remove old dat symlink 
 if(remove("sparseConVec.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
 else { puts( "File successfully deleted" ); }
 fpSparseConVec = fopen("sparseConVec.dat", "wb");
 nElementsWritten = fwrite(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
 fclose(fpSparseConVec);
 if(nElementsWritten != nConnections) {
   printf("\n Error: All elements not written \n");
 }
 
 printf("writing new cons to file ... "); fflush(stdout);
 fpIsRewiredLink = fopen("newPostNeurons.dat", "wb");
 nElementsWritten = 0;
 nElementsWritten = fwrite(IS_REWIRED_LINK, sizeof(*IS_REWIRED_LINK), nConnections, fpIsRewiredLink);
 fclose(fpIsRewiredLink);
 if(nElementsWritten != nConnections) {
   printf("\n Error: All elements not written \n");
 }
 
 if(remove("idxVec.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
 else { puts( "File successfully deleted" ); }
  fpIdxVec = fopen("idxVec.dat", "wb");
  fwrite(idxVec, sizeof(*idxVec), N_NEURONS,  fpIdxVec);
 fclose(fpIdxVec);
 if(remove("nPostNeurons.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
 else { puts( "File successfully deleted" ); }
 fpNpostNeurons = fopen("nPostNeurons.dat", "wb");
 fwrite(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
 fclose(fpNpostNeurons);
 printf("done\n");
 delete [] poOfNeurons;
 // vector<unsigned long> nPost(NE);
 for(i = 0; i < NE; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < NE; j++) {
   for(i = 0; i < NE; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
     }
   }
 }
 FILE *ffp22;
 ffp22 = fopen("rewired_K_EE_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp22, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp22);
 delete [] IF_REWIRED_CON;

}

void RemoveConnections(unsigned int *conVec, double kappa) {
 unsigned long long int i, j;
 double removalProb = kappa / sqrt((double)K);
 printf("removal prob = %f\n", removalProb);
 std::random_device rd;
 std::default_random_engine gen(rd());
 std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
 vector<unsigned long> nPost(NE);
 // COUNT BEFORE REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < NE; j++) {
   for(i = 0; i < NE; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
     }
   }
 }
 FILE *ffp;
 ffp = fopen("orig_K_EE_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp); 
 // REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   for(j = 0; j < NE; j++) {
     if(conVec[i + N_NEURONS * j]) {
       if(removalProb >= UniformRand(gen)) {
	 conVec[i + N_NEURONS * j]  = 0;
       }
     }
   }
 }
 // COUNT AFTER REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < NE; j++) {
   for(i = 0; i < NE; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
     }
   }
 }
 ffp = fopen("removed_K_EE_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp); 
}


void GetFullMat(unsigned int *conVec, unsigned int* sparseVec, unsigned int* idxVec, unsigned int* nPostNeurons) {
 unsigned long long int i, j, k, nPost;
 // INITIALIZE
 for(i = 0; i < N_NEURONS; i++) {
   for(j = 0; j < N_NEURONS; j++) {
     conVec[i + N_NEURONS * j]  = 0;
   }
 }

 //  for(i = 0; i < N_NEURONS; i++) {
 //   for(j = 0; j < N_NEURONS; j++) {
 //     printf("%d ", (int)conVec[i + N_NEURONS * j]);
 //   }
 //   printf("\n");
 // }

  
 // CONSTRUCT
 for(i = 0; i < N_NEURONS; i++) {
   nPost = nPostNeurons[i];
   for(k = 0; k < nPost; k++) {
     j = sparseVec[idxVec[i] + k];
     conVec[i + N_NEURONS * j]  = 1; /* i --> j  */
    }
  }


 // printf("----------------------------------------\n");
 // printf("--------- RECONSTRUCTED MATRIX ---------\n");
 // for(i = 0; i < N_NEURONS; i++) {
 //   for(j = 0; j < N_NEURONS; j++) {
 //     printf("%d ", (int)conVec[i + N_NEURONS * j]);
 //   }
 //   printf("\n");
 // }

}

void GenConMat(int EE_CON_TYPE) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
  //  int dummy;
  unsigned long long int nConnections = 0;
  cout << "generating conmat" << endl;
  conMat = new unsigned int [(unsigned long int)N_NEURONS * N_NEURONS];
  double *poOfNeurons = new double [NE];  
  ///////////////////////// READ POs  ///////////////////////////////////
  if(not EE_CON_TYPE) {
    FILE *fpPOofNeurons;
    fpPOofNeurons = fopen("poOfNeurons.dat", "rb");
    unsigned long int nPOsRead = 0;
    nPOsRead = fread(poOfNeurons, sizeof(*poOfNeurons), NE, fpPOofNeurons);
    if(nPOsRead != NE) {
      printf("\n Error: All POs not read\n");
    }  
    fclose(fpPOofNeurons);
    // printf("POs :\n");
    // for (unsigned long int i = 0; i < 10; i++)  {
    //   printf("%.4f ", poOfNeurons[i]);
    // }
    // printf("\n");
  }
  //////////////////////////////////////////////////////////////////////
  
  for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    for (unsigned long int j = 0; j < N_NEURONS; j++)  {
      conMat[i + N_NEURONS * j] = 0;
    }
  }

  cout << SQRT_K << " " << recModulationEE << endl;
  for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    for (unsigned long int j = 0; j < N_NEURONS; j++)  {
      // i --> j

      if(i < NE && j < NE) { //E-to-E
	double conProbEE = ConProb(i * M_PI / (double)NE, j * M_PI / (double)NE, NE, recModulationEE);
        if(not EE_CON_TYPE) {
	  cout << "not EE_CON_TYPE" << endl;
	  conProbEE = (double)K * (1 + 2.0 * kappa * cos(2.0 * (poOfNeurons[i] - poOfNeurons[j])) / sqrt((double)K) ) / (double)NE;
	}
	if(UniformRand(gen) <= conProbEE) {
	  conMat[i + N_NEURONS * j] = 1;
	  nConnections += 1;
	}
      }

      if(i < NE && j >= NE) { //E-to-I
	if(UniformRand(gen) <= ConProb(i * M_PI / (double)NE, j * M_PI / (double)NI, NE, recModulationIE)) {
	  conMat[i + N_NEURONS * j] = 1;
	  nConnections += 1;
	}
      }
      if(i >= NE && j < NE) { //I-to-E
	if(UniformRand(gen) <= ConProb(i * M_PI / (double)NI, j * M_PI / (double)NE, NI, recModulationEI)) {
	  conMat[i + N_NEURONS * j] = 1;
	  nConnections += 1;
	}
      }
      if(i >= NE && j >= NE) { //I-to-I
	if(UniformRand(gen) <= (double)K / (double)NI) {
	  conMat[i + N_NEURONS * j] = 1;
	  nConnections += 1;
	}
      }
    }
  }


  
  cout << "done" << endl;

  // cout << "computing moments of conmat" << endl;
  // double C0 = 0;
  // double C1 = 0;
  // double ddPhi = M_PI / (double)NE;
  
  // for (unsigned long int i = 0; i < NE; i++)  {
  //   double xCord = 0, yCord = 0;
  //   for (unsigned long int j = 0; j < NE; j++)  {
  //     C0 += conMat[i + N_NEURONS * j] / (double) NE;
  //     xCord += conMat[i + N_NEURONS * j] * cos(2.0 * j * ddPhi);
  //     yCord += conMat[i + N_NEURONS * j] * sin(2.0 * j * ddPhi);
  //   }
  //   C1 += (1.0 / (double)NE) * sqrt(xCord * xCord + yCord * yCord);
  // }

  // cout << "C0 = " << C0 << " C1 = " << C1 << endl;


  
 // // COUNT AFTER ADDING CONNECTIONS
 // vector<unsigned long> nPost(NE);  
 // for(unsigned long int i = 0; i < NE; i++) {
 //   nPost[i] = 0;
 // } 
 // for(unsigned long int j = 0; j < NE; j++) {
 //   for(unsigned long int i = 0; i < NE; i++) {
 //     if(conMat[i + NE * j]) {
 //       nPost[j] += 1;
 //       nConnections += 1;
 //     }
 //   }
 // }
 // FILE *ffptmp = fopen("FULL_COS_K_count.csv", "w");
 // for(unsigned int lll = 0; lll < NE; lll++) {
 //   fprintf(ffptmp, "%lu\n", (unsigned long)nPost[lll]);
 // }
 // fclose(ffptmp);
 // nPost.clear();

 // printf("----------------------------------------\n");
 // printf("------------ ORIGINAL MATRIX -----------\n");
 // for(unsigned long int i = 0; i < N_NEURONS; i++) {
 //   for(unsigned long int j = 0; j < N_NEURONS; j++) {
 //     printf("%d ", (int)conMat[i + N_NEURONS * j]);
 //   }
 //   printf("\n");
 // }
 // printf("----------------------------------------\n");
 // printf("----------------------------------------\n"); 


  

  for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    for (unsigned long int j = 0; j < N_NEURONS; j++)  {
      if((conMat[i + N_NEURONS * j] < 0) || (conMat[i + N_NEURONS * j] > 1)) {
  	  printf("%lu %lu !!!!!!! oh la la !!!!\n", i, j);
  	}
    }
  }

  
  unsigned long int nElementsWritten;
  // FILE *fpcmat;
  // fpcmat = fopen("cmat.dat", "wb");
  // nElementsWritten = fwrite(conMat, sizeof(unsigned int), N_NEURONS * N_NEURONS, fpcmat);
  // fclose(fpcmat);

  // for (unsigned long int i = 0; i < NE; i++)  {
  //   printf("\n");    
  //   for (unsigned long int j = 0; j < NE; j++)  {
  //     printf("%u ", conMat[i + N_NEURONS * j]);
  //     // printf("%lu: %u\n", i + N_NEURONS * j, conMat[i + N_NEURONS * j]);
  //   }
  // }
  // printf("\n\n");


  
  cout << "computing sparse rep" << endl;    
  sparseConVec = new unsigned int[nConnections];
  IS_REWIRED_LINK = new unsigned int[nConnections];
  for(unsigned long i = 0; i < nConnections; i++) {
    IS_REWIRED_LINK[i] = 0;
  }
  GenSparseMat(conMat, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  cout << "done" << endl;
  FILE *ffp;
  ffp = fopen("kcount.csv", "w");
  for(unsigned int lll = 0; lll < N_NEURONS; lll++) {
    fprintf(ffp, "%u\n", nPostNeurons[lll]);
  }
  fclose(ffp);
  //  delete [] conMat;
  if(not EE_CON_TYPE) {
    delete [] poOfNeurons;
  }
  // write to file
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  printf("done\n#connections = %llu\n", nConnections);
  printf("writing to file ... "); fflush(stdout);
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  nElementsWritten = fwrite(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  if(nElementsWritten != nConnections) {
    printf("\n Error: All elements not written \n");
  }
  fpIdxVec = fopen("idxVec.dat", "wb");
  fwrite(idxVec, sizeof(*idxVec), N_NEURONS,  fpIdxVec);
  fclose(fpIdxVec);
  fpNpostNeurons = fopen("nPostNeurons.dat", "wb");
  fwrite(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");
  // for (unsigned long int i = 0; i < 10; i++)  {
  //     printf("%u ", sparseConVec[i]);
  // }
  // exit(1);  
}

void VectorSum(vector<double> &a, vector<double> &b) { 
  for(unsigned int i = 0; i < N_NEURONS; ++i) {
    a[i] += b[i];
  }
}

void VectorSumFF(vector<double> &a, vector<double> &b) { 
  for(unsigned int i = 0; i < NFF; ++i) {
    a[i] += b[i];
  }
}

void VectorDivide(vector<double> &a, double z) { 
   for(unsigned int i = 0; i < N_NEURONS; ++i) {
     a[i] /= z;
   }
}

void VectorDivideFF(vector<double> &a, double z) { 
   for(unsigned int i = 0; i < NFF; ++i) {
     a[i] /= z;
   }
}


void VectorCopy(vector<double> &a, vector<double> &b) {
  // copy elements of a to b
   for(unsigned int i = 0; i < N_NEURONS; ++i) {
     b[i] = a[i];
   }
}


double PopAvg(vector<double> &a, unsigned int start, unsigned int end) {
  double result = 0.0;
  for(unsigned int i = start; i < end; ++i) {
    result += a[i];
  }
  return result / (double)(end - start);
}

double AddElements(vector<double> &a, unsigned int start, unsigned int end) {
  // add elements of a[start] to a[end]
  double result = 0.0;
  for(unsigned int i = start; i < end; ++i) {
    result += a[i];
  }
  return result;
}

double Heavside(double input) {
  if(input > 0) {
    return 1.0;
  }
  else {
    return 0.0;
  }
}

void Create_Dir(string dir) {
  string mkdirp = "mkdir -p " ;
  mkdirp += dir;
  const char * cmd = mkdirp.c_str();
  const int dir_err = system(cmd);
  if(-1 == dir_err) {
    cout << "error creating directories" << endl ;
  }
  cout << "Created directory : " ;
  cout << mkdirp << endl ;
}

void LoadSparseConMat() {
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  //  int ;
  unsigned long int nConnections = 0, dummy = 0;
  printf("%p %p %p\n", fpIdxVec, fpNpostNeurons, fpSparseConVec);
  dummy = fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  for(unsigned int i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  printf("#Post read\n #rec cons = %lu \n", nConnections);  
  sparseConVec = new unsigned int[nConnections] ;
  dummy = fread(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  if(dummy != nConnections) {
    printf("sparseConvec read error ? \n");
  }
  printf("#sparse cons = %lu \n\n", dummy);    
  printf("sparse vector read\n");  
  dummy = fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  printf("#idx vector read\n");    
  fclose(fpSparseConVec);
  fclose(fpIdxVec);
}

void LoadRewiredCon(unsigned long nElements) {
  unsigned long nElementsRead = 0;
  FILE *fpRewired;
  fpRewired = fopen("newPostNeurons.dat", "rb");
  nElementsRead = fread(IS_REWIRED_LINK, sizeof(*IS_REWIRED_LINK), nElements, fpRewired);
  printf("reading new post neurons\n");
  if(nElements != nElementsRead) {
    printf("rewired read error ? \n");
  }
  printf("done\n");
}

void LoadFFSparseConMat() {
  FILE *fpSparseConVecFF, *fpIdxVecFF, *fpNpostNeuronsFF;
  //  fpSparseConVecFF = fopen("sparseConVecFF.dat", "rb");
  perror("ohlala");
  fpIdxVecFF = fopen("idxVecFF.dat", "rb");
  fpNpostNeuronsFF = fopen("nPostNeuronsFF.dat", "rb");
  unsigned long int long nConnections = 0, dummy = 0;
  // printf("%p %p %p\n", fpIdxVecFF, fpNpostNeuronsFF, fpSparseConVecFF);
  dummy = fread(nPostNeuronsFF, sizeof(*nPostNeuronsFF), NFF, fpNpostNeuronsFF);
  fclose(fpNpostNeuronsFF);
  printf("#Post read\n");
  for(unsigned int i = 0; i < NFF; ++i) {
    nConnections += nPostNeuronsFF[i];
  }
  printf("#connections = %llu\n", nConnections);  
  sparseConVecFF = new unsigned int[nConnections];
  fpSparseConVecFF = fopen("sparseConVecFF.dat", "rb");
  perror("lalala");
  dummy = fread(sparseConVecFF, sizeof(*sparseConVecFF), nConnections, fpSparseConVecFF);
  perror("ooplala");
  //    dummy = fread(sparseConVecFF, sizeof(*sparseConVecFF), 1, fpSparseConVecFF);
  printf("sparseConvec read: %llu %llu \n", dummy, nConnections);
  if(dummy != nConnections) {
    printf("sparseConvec read error ? %llu %llu \n", dummy, nConnections);
  }
  printf("sparse vector read\n");  
  dummy = fread(idxVecFF, sizeof(*idxVecFF), NFF, fpIdxVecFF);
  printf("#idx vector read\n");    
  fclose(fpSparseConVecFF);
  fclose(fpIdxVecFF);
}

void RunSim() {
  double dt, probUpdateFF, probUpdateE, probUpdateI, uNet, spinOld = 0, spinOldFF = 0;
  // uExternalE, uExternalI;
  unsigned long int nSteps, i, nLastSteps, nInitialSteps, intervalLen = (NE + NI) / 100;
  unsigned int updateNeuronIdx = 0, chunkIdx = 0;
  int updatePop = 0; // 0: FF, 1: E, 2: I
  double runningFre = 0, runningFri = 0, runningFrFF = 0;
  // FILE *fpMeanRates;
  vector<double> spins(N_NEURONS);
  vector<double> spinsFF(NFF);  
  vector<double> spkTimes;
  vector<unsigned long int> spkNeuronIdx;
  vector<double> firingRates(N_NEURONS);
  vector<double> firingRatesFF(NFF);  
  vector<double> frLast(N_NEURONS);
  vector<double> totalInput(N_NEURONS);
  vector<double> FFInput(N_NEURONS);  
  vector<double> netInputVec(N_NEURONS);
  vector<double> inputVecE(N_NEURONS); // recurrent E input u_E = u_net - u_FF - u_I
  vector<double> inputVecI(N_NEURONS); // recurrent I input u_I = u_net - u_FF - u_E
  vector<double> firingRatesChk(N_NEURONS);
  vector<double> firingRatesChkTMP(N_NEURONS);
  vector<double> firingRatesAtT(N_NEURONS);  
  vector<double> ratesAtInterval(N_NEURONS);
  double popME1, popME1Phase;  // popMI1, popMI1Phase,
  double phiExtOld = phi_ext;
  std::string m1FileName;     
  m1FileName = "MI1_inst_theta" + std::to_string(phi_ext * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + ".txt";
  FILE *fpInstM1 = fopen(m1FileName.c_str(), "w");  
  dt = (TAU_FF * TAU_E * TAU_I) / (NE * TAU_FF * TAU_I + NI * TAU_FF * TAU_E + NFF * TAU_E * TAU_I);
  nSteps = (unsigned long)((tStop / dt) + (T_TRANSIENT / dt));
  nInitialSteps = (T_TRANSIENT / dt);
  probUpdateFF = dt * (double)NFF / TAU_FF;  
  probUpdateE = dt * (double)NE / TAU_E;
  probUpdateI = dt * (double)NI / TAU_I;
  // uExternalE = sqrt((double) K) * JE0 * m0;
  // uExternalI = sqrt((double) K) * JI0 * m0;
  // printf("%f\n", uExternalE);
  // printf("%f\n", uExternalI);
  if(IF_STEP_PHI0) {
    // used for computing m_E^1(t)
    intervalLen = (unsigned long)floor((nSteps - nInitialSteps) / 80.0);
  }

////////////////////////////////////////////////////////////////////////////////
  // srand (time(NULL));
  // for(unsigned l = 0; l < N_NEURONS; l++) {
  //   frLast[l] = 0;
  //   spins[l] = 0;
  //   spinsFF[l] = 0;
  //   if(UniformRand() < 0.5) {
  //     spins[l] = 1;
  //   }
  //   unsigned int tmpIdxInit, cntrInit;
  //   if(spins[l]) {
  //     cntrInit = 0;
  //     tmpIdxInit = idxVec[l];
  //     while(cntrInit < nPostNeurons[l]) { // update all the post neurons
  // 	unsigned int kkInit = sparseConVec[tmpIdxInit + cntrInit];
  // 	cntrInit += 1;
  // 	if(l < NE) {
  // 	  if(kkInit < NE) {
  // 	    netInputVec[kkInit] += JEE_K;
  // 	  }
  // 	  else {
  // 	    netInputVec[kkInit] += JIE_K;
  // 	  }	    
	    
  // 	}
  // 	else {
  // 	  if(kkInit < NE) {
  // 	    netInputVec[kkInit] += JEI_K;
  // 	  }
  // 	  else {
  // 	    netInputVec[kkInit] += JII_K;
  // 	  }
  // 	}
  //     }
  //   }
  // }
  
////////////////////////////////////////////////////////////////////////////////  

  
    
  
  printf("dt = %f, #steps = %lu, T_STOP = %d\n", dt, nSteps, T_STOP);


  
  printf("prob updateFF = %f, prob update E = %f, prob update I = %f\n", probUpdateFF, probUpdateE, probUpdateI);
  nLastSteps = nSteps - (unsigned long int )((float)T_TRANSIENT / dt);
  printf("\n");
  ProgressBar(0.0, runningFre, runningFri, runningFrFF);
  FILE *fpInstRates = fopen("fr_inst.txt", "w");
  std::random_device rdFF;  
  std::default_random_engine generatorFF(rdFF());
  std::discrete_distribution<int> MultinomialDistr {probUpdateFF, probUpdateE, probUpdateI};
  int nPhisInSteps = 2;
  for(i = 0; i < nSteps; i++) {
    if(IF_STEP_PHI0) {
      if((i > 0) && i % (unsigned long)floor((nSteps - nInitialSteps) / (double)nPhisInSteps) == 0) {
	// change stimulus after half time
	phi_ext = phi_ext +  0.5 * M_PI / (double)nPhisInSteps;
	// printf("hello %f\n", phi_ext * 180  / M_PI);      
      }
    }
    if(i > 0 && i % (nSteps / 100) == 0) {
      runningFrFF = PopAvg(firingRatesFF, 0, NFF) / (double)(i + 1);      
      runningFre = PopAvg(firingRates, 0, NE) / (double)(i + 1);
      runningFri = PopAvg(firingRates, NE, N_NEURONS) / (double)(i + 1);
      fprintf(fpInstRates, "%f;%f\n", runningFre, runningFri);
      fflush(fpInstRates);
      ProgressBar((float)i / nSteps, runningFre, runningFri, runningFrFF);    
    }
    uNet = 0.0;
    updatePop = MultinomialDistr(generatorFF);
    if(updatePop == 0) {
      updateNeuronIdx = RandFFNeuron();
      spinOldFF = spinsFF[updateNeuronIdx];
      spinsFF[updateNeuronIdx] = 0;
      if(UniformRand() <= FFTuningCurve(updateNeuronIdx, phi_ext)) {
        // FFTuning returons prob of state_i = 1	
	spinsFF[updateNeuronIdx] = 1;
      }

//////////////////////////////////////////////////
      // FF input
      // if(i >= nLastSteps) { 
      // 	unsigned int tmpIdx1, cntr1, iidxx;
      // 	cntr = 0;
      // 	for(iidxx = 0; iidxx < NFF; iidxx++) {
      // 	  if(spinsFF[iidxx]) {
      // 	    tmpIdx1 = idxVecFF[iidxx];
      // 	    while(cntr < nPostNeuronsFF[iidxx]) {
      // 	      unsigned int kk = sparseConVecFF[tmpIdx1 + cntr];
      // 	      cntr1 += 1;
      // 	      if(kk < NE) {
      // 		FFInput[kk] += JE0_K;
      // 	      }
      // 	      else {
      // 		FFInput[kk] += JI0_K;
      // 	      }
      // 	    }
      // 	  }
      // 	}
      // }
//////////////////////////////////////////////////      

      
      
      if(spinOldFF == 0 && spinsFF[updateNeuronIdx] == 1) {
	unsigned int tmpIdx, cntr;
	cntr = 0;
	tmpIdx = idxVecFF[updateNeuronIdx];
	while(cntr < nPostNeuronsFF[updateNeuronIdx]) {
	  unsigned int kk = sparseConVecFF[tmpIdx + cntr];
	  cntr += 1;
	  if(kk < NE) {
	    netInputVec[kk] += JE0_K;
	    // if(i >= nLastSteps) { FFInput[kk] += JE0_K;}
	  }
	  else {
	    netInputVec[kk] += JI0_K;
	  }
	}
      }
      else if(spinOldFF == 1 && spinsFF[updateNeuronIdx] == 0) {
	unsigned int tmpIdx, cntr = 0;
	cntr = 0;      
	tmpIdx = idxVecFF[updateNeuronIdx];
	while(cntr < nPostNeuronsFF[updateNeuronIdx]) {
	  unsigned int kk = sparseConVecFF[tmpIdx + cntr];
	  cntr += 1;
	  if(kk < NE) {
	    netInputVec[kk] -= JE0_K;
	    //if(i >= nLastSteps) { FFInput[kk] -= JE0_K;}
	  }
	  else {
	    netInputVec[kk] -= JI0_K;
	  }
	}
      }
    }
    else {
      if(updatePop == 1) {
	updateNeuronIdx = RandENeuron();
	uNet = netInputVec[updateNeuronIdx] - THRESHOLD_E;
      }
      else if(updatePop == 2)  {
	updateNeuronIdx = RandINeuron();
	uNet = netInputVec[updateNeuronIdx] - THRESHOLD_I;     
      }
      spinOld = spins[updateNeuronIdx];    
      spins[updateNeuronIdx] = Heavside(uNet);
      if(spinOld == 0 && spins[updateNeuronIdx] == 1) {
	unsigned int tmpIdx, cntr;
	cntr = 0;
	tmpIdx = idxVec[updateNeuronIdx];
	while(cntr < nPostNeurons[updateNeuronIdx]) {
	  unsigned int kk = sparseConVec[tmpIdx + cntr];
	  cntr += 1;
	  if(updateNeuronIdx < NE)  {
	    unsigned int IS_STRENGTHENED = 0;
	    if(IF_LOADREWIREDCON) {
	      IS_STRENGTHENED = IS_REWIRED_LINK[tmpIdx + cntr - 1];
	    }
	    if(kk < NE) {
	      if(IS_STRENGTHENED) {
		netInputVec[kk] += (rewiredEEWeight * JEE_K);
		// printf("rewired synapse!\n");
	      }
	      else {
		netInputVec[kk] += JEE_K;
	      }
	    }
	    else {
	      netInputVec[kk] += JIE_K;
	    }
	  }
	  else {
	    if(kk < NE) {
	      netInputVec[kk] += JEI_K;
	    }
	    else {
	      netInputVec[kk] += JII_K;
	    }
	  }
	}
      }
      else if(spinOld == 1 && spins[updateNeuronIdx] == 0) {
	// unsigned int tmpIdx;
	// tmpIdx = idxVec[updateNeuronIdx];
	// for(unsigned int kk = sparseConVec[tmpIdx]; kk < sparseConVec[tmpIdx + nPostNeurons[updateNeuronIdx]]; kk++) {
	unsigned int tmpIdx, cntr = 0;
	cntr = 0;      
	tmpIdx = idxVec[updateNeuronIdx];
	while(cntr < nPostNeurons[updateNeuronIdx]) {
	  unsigned int kk = sparseConVec[tmpIdx + cntr];
	  cntr += 1;
	  if(updateNeuronIdx < NE)  {
	    unsigned int IS_STRENGTHENED = 0;
	    if(IF_LOADREWIREDCON) {
	      IS_STRENGTHENED = IS_REWIRED_LINK[tmpIdx + cntr - 1];
	    }
	    if(kk < NE) {
	      if(IS_STRENGTHENED) {
		netInputVec[kk] -= (rewiredEEWeight * JEE_K);
	      }
	      else {
		netInputVec[kk] -= JEE_K;
	      }
	    }
	    else {
	      netInputVec[kk] -= JIE_K;
	    }
	  }
	  else {
	    if(kk < NE) {
	      netInputVec[kk] -= JEI_K;
	    }
	    else {
	      netInputVec[kk] -= JII_K;
	    }
	  }
	}
      }
    }

    VectorSum(ratesAtInterval, spins);           
    if(i > 0 && (i % intervalLen == 0)){
      VectorDivide(ratesAtInterval, (double)intervalLen);
      M1ComponentE(ratesAtInterval, NE, &popME1, &popME1Phase);
      	 
      if(IF_STEP_PHI0) {
	fprintf(fpInstM1, "%f;%f;%f\n", popME1, popME1Phase, phi_ext);
        fflush(fpInstM1);
      }
      else {
	fprintf(fpInstM1, "%f;%f\n", popME1, popME1Phase);
	fflush(fpInstM1);
      }
      for(unsigned int ijk = 0; ijk < N_NEURONS; ijk++) {
	ratesAtInterval[ijk] = 0;
      }
    }
    
    if(spinOld == 0 && spins[updateNeuronIdx] == 1) {
      spkTimes.push_back(i * dt);
      spkNeuronIdx.push_back(updateNeuronIdx);
    }
    
   VectorSum(firingRates, spins);
   VectorSumFF(firingRatesFF, spinsFF);   
   if(i >= nLastSteps) {
     VectorSum(frLast, spins);
     VectorSum(totalInput, netInputVec);
   }

   //   Store Chunks 
   if(i >= nInitialSteps) {
     VectorSum(firingRatesChk, spins);
     if(i > nInitialSteps) {
       if(!((i - nInitialSteps) %  (unsigned int)((nSteps - nInitialSteps - 1) / (double)N_SEGMENTS))) {
	 double chunckLength = (double)((nSteps - nInitialSteps - 1) / (double)N_SEGMENTS);
	 // if(!((i - nInitialSteps) %  (unsigned int)((nSteps - nInitialSteps - 1) / 2))) {	 
	 printf("\n chk i = %lu \n", i - nInitialSteps);
	 VectorCopy(firingRatesChk, firingRatesChkTMP);
	 // VectorDivide(firingRatesChkTMP, (i - nInitialSteps));
	 VectorDivide(firingRatesChkTMP, chunckLength);
	 std::string txtFileName = "meanrates_theta" + std::to_string(phi_ext * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_chnk" + std::to_string(chunkIdx) + ".txt";
	 chunkIdx += 1;       
	 FILE *fpRates = fopen(txtFileName.c_str(), "w");
	 for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
	   fprintf(fpRates, "%f\n", firingRatesChkTMP[ii]);
	 }
	 fclose(fpRates);
	 for(unsigned int ijk = 0; ijk < N_NEURONS; ijk++) {
	   firingRatesChk[ijk] = 0;
	 }
	 
       }
     }
   }
  }
  // COMPUTE RATES: NORMALIZE BY TIME    
  VectorDivide(firingRates, nSteps);
  VectorDivideFF(firingRatesFF, nSteps);  
  VectorDivide(frLast, (nSteps - nLastSteps));
  VectorDivide(totalInput, (nSteps - nLastSteps));
  // VectorDivide(FFInput, (nSteps - nLastSteps));    
  fclose(fpInstRates);
  fclose(fpInstM1);
  // ESTIMATE FF INPUT
  for(unsigned int lll = 0; lll < N_NEURONS; lll++) {
    inputVecE[lll] = 0;
    inputVecI[lll] = 0;
    if(lll < NFF) { FFInput[lll] = 0; }
  }  
  unsigned int tmpIdx1, cntr1, iidxx;
  for(iidxx = 0; iidxx < NFF; iidxx++) {
    tmpIdx1 = idxVecFF[iidxx];
    cntr1 = 0;      
    while(cntr1 < nPostNeuronsFF[iidxx]) {
      unsigned int kk = sparseConVecFF[tmpIdx1 + cntr1];
      cntr1 += 1;
      if(kk < NE) {
	FFInput[kk] += JE0_K * firingRatesFF[iidxx];
      }
      else {
	FFInput[kk] += JI0_K * firingRatesFF[iidxx];
      }
    }
  }
  // ESTIMATE recurrent u_EE
  for(iidxx = 0; iidxx < NE; iidxx++) { // if pre-neurons are E
    tmpIdx1 = idxVec[iidxx];
    cntr1 = 0;      
    while(cntr1 < nPostNeurons[iidxx]) {
      unsigned int kk = sparseConVec[tmpIdx1 + cntr1];
      cntr1 += 1;
      if(kk < NE) { // if post-neuron is E
  	unsigned int IS_STRENGTHENED = 0;
  	if(IF_LOADREWIREDCON) { IS_STRENGTHENED = IS_REWIRED_LINK[tmpIdx1 + cntr1 - 1]; }
  	if(IS_STRENGTHENED) { inputVecE[kk] += JEE_K * rewiredEEWeight * firingRates[iidxx]; }
  	else { inputVecE[kk] += JEE_K * firingRates[iidxx]; }
      }
      if(kk >= NE) { // if post-neuron is I 
  	inputVecE[kk] += JIE_K * firingRates[iidxx];
      }      
    }
  }
  // ESTIMATE recurrent u_EI  
  for(iidxx = NE; iidxx < N_NEURONS; iidxx++) { // if pre-neurons are I
    tmpIdx1 = idxVec[iidxx];
    cntr1 = 0;      
    while(cntr1 < nPostNeurons[iidxx]) {
      unsigned int kk = sparseConVec[tmpIdx1 + cntr1];
      cntr1 += 1;
      // if post-neuron is E
      if(kk < NE) { inputVecI[kk] += JEI_K * firingRates[iidxx]; }
      // if post-neuron is I       
      if(kk >= NE) { inputVecI[kk] += JII_K * firingRates[iidxx]; }            
    }
  }
  // // ESTIMATE 
  // for(iidxx = 0; iidxx < NE; iidxx++) { // if pre-neurons are E
  //   tmpIdx1 = idxVec[iidxx];
  //   cntr1 = 0;      
  //   while(cntr1 < nPostNeurons[iidxx]) {
  //     unsigned int kk = sparseConVec[tmpIdx1 + cntr1];
  //     cntr1 += 1;
  //     if(kk < NE) { // if post-neuron is E
  // 	unsigned int IS_STRENGTHENED = 0;
  // 	if(IF_LOADREWIREDCON) { IS_STRENGTHENED = IS_REWIRED_LINK[tmpIdx1 + cntr1 - 1]; }
  // 	if(IS_STRENGTHENED) { inputVecE[kk] += JEE_K * rewiredEEWeight * firingRates[iidxx]; }
  // 	else { inputVecE[kk] += JEE_K * firingRates[iidxx]; }
  //     }
  //   }
  // }
  
  //
  std::string txtFileName = "meanrates_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + ".txt";  
  FILE *fpRates = fopen(txtFileName.c_str(), "w");
  for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
    fprintf(fpRates, "%f\n", firingRates[ii]);
  }
  fclose(fpRates);

  txtFileName = "meanrates_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_last.txt";  
  fpRates = fopen(txtFileName.c_str(), "w");
  for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
    fprintf(fpRates, "%f\n", frLast[ii]);
  }
  fclose(fpRates);

  if(IF_SAVE_INPUT) {
    txtFileName = "meaninput_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_last.txt";  
    FILE *fpInputs = fopen(txtFileName.c_str(), "w");
    for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
      fprintf(fpInputs, "%f\n", totalInput[ii]);
    }
    fclose(fpInputs);
    // save u_EE
    txtFileName = "meaninput_E_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_last.txt";  
    fpInputs = fopen(txtFileName.c_str(), "w");
    for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
      fprintf(fpInputs, "%f\n", inputVecE[ii]);
    }
    fclose(fpInputs);    
    // SAVE u_EI
    txtFileName = "meaninput_I_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_last.txt";  
    fpInputs = fopen(txtFileName.c_str(), "w");
    for(unsigned int ii = 0; ii < N_NEURONS; ii++) {
      fprintf(fpInputs, "%f\n", inputVecI[ii]);
    }
    fclose(fpInputs);    
    // save FF
    txtFileName = "meanFFinput_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + "_last.txt";  
    FILE *fpFFInputs = fopen(txtFileName.c_str(), "w");
    for(unsigned int ii = 0; ii < NE+NI; ii++) {
      // fprintf(fpFFInputs, "%f\n", firingRatesFF[ii]);
      fprintf(fpFFInputs, "%f\n", FFInput[ii]);      
    }
    fclose(fpFFInputs);
  }

  
  if(IF_SAVE_SPKS) {
      txtFileName = "spktimes_theta" + std::to_string(phiExtOld * 180 / M_PI) + "_tr" + std::to_string(trialNumber) + ".txt";
      FILE *fpSpks = fopen(txtFileName.c_str(), "w");
      for(unsigned long long int ii = 0; ii < spkNeuronIdx.size(); ii++) {
	fprintf(fpSpks, "%lu;%f\n", spkNeuronIdx[ii], spkTimes[ii]);
      }
      fclose(fpSpks);
  }

  spins.clear();
  spkTimes.clear();
  spkNeuronIdx.clear();
  firingRates.clear();
  totalInput.clear();
  FFInput.clear();
  netInputVec.clear();
  firingRatesChk.clear();
  firingRatesAtT.clear();  
  ratesAtInterval.clear();
  printf("\nsee you later, alligator\n");
}
  
int main(int argc, char *argv[]) {
  
  printf("#args = %d\n", argc);
  if(argc > 1) {
    m0_ext = atof(argv[1]);
  }
  if(argc > 2) {
    printf("hello mExtOn1\n");
    m1_ext = atof(argv[2]);
  }
  if(argc > 3) {
    recModulationEE = atof(argv[3]); // parameter p // absolete remove this p is now kappa!!!
    kappa = recModulationEE;
    recModulationIE = 0; //-1.0 * recModulationEE;
    recModulationEI = 0; //-1.0 * recModulationEE;
  }
  if(argc > 4) {
    ffModulation = atof(argv[4]); // parameter gamma
  }
  if(argc > 5) {
    phi_ext = M_PI * atof(argv[5]) / 180.0; // parameter external orientation
  }
  if(argc > 6) {
    trialNumber = atof(argv[6]);
  }
  if(argc > 7) {
    kappa = atof(argv[7]); // this is redundant!!!! clean up!@#$!
    kappa = recModulationEE;
  }
  if(argc > 8) {
    if(IF_REWIRE == 1) {
      rewiredEEWeight = atof(argv[8]);
    }
    else {
      rewiredEEWeight = 1;
      // recModulationEE = kappa;
    }
  }
  if(argc > 9) {
    IF_GEN_MAT = atoi(argv[9]);
  }


  double dt=1e-5;
  dt = (TAU_FF * TAU_E * TAU_I) / (NE * TAU_FF * TAU_I + NI * TAU_FF * TAU_E + NFF * TAU_E * TAU_I);
  printf("dt = %f, #steps = %lu, T_STOP = %f\n", dt, nSteps, T_STOP);

  exit(1);

  // std::default_random_engine genFFDet(1234U);
  // std::uniform_real_distribution<double> UniformRand_det(0.0, 1.0);
  // for(int i = 0; i < 10; i++) {
  //   double poFFi = M_PI * UniformRand_det(genFFDet);
  //   printf("gen po = %f \n", poFFi);
  // }
  // exit(1);

  
  tStop = T_STOP;

  cout << "NE = " << NE << " NI = " << NI << " NFF = " << NFF << " K = " << K << " p = " << atof(argv[3]) << " m0 = " << m0_ext << " m0_One = " << m1_ext << endl;
  cout << "gamma = " << ffModulation << " KFF = " << cFF * K << " Phi_Ext = " << phi_ext * 180.0 / M_PI << endl;
  cout << "TAU_FF = " << TAU_FF << " TAU_E = " << TAU_E << " Tau_I = " << TAU_I << endl;
  cout << "Trial# = " << trialNumber << endl;
  cout << "ji_factor = " << JI_FACTOR << endl;
  cout << "kappa = " << kappa << endl;

  
  
  //  sprintf(folderName, "N%uK%um0%dpgamma%dT%d", N_NEURONS, (int)(m0 * 1e3), K, recModulation, ffModulation, (int)(tStop * 1e-3));
  // folderName = "./data/N" + std::to_string(N_NEURONS) + "K" + std::to_string(K) + "m0" + std::to_string((int)(m0 * 1e3)) + "p" + std::to_string((unsigned int)(10 * recModulation)) + "gamma" + std::to_string((unsigned int)(ffModulation)) + std::to_string((int)(tStop * 1e-3));

  // folderName = "./data/N" + std::to_string(N_NEURONS) + "K" + std::to_string(K) + "m0" + std::to_string((int)(m0 * 1e3)) + "p" + std::to_string((unsigned int)(10 * recModulationEE)) + "gamma0" + std::to_string((int)(tStop * 1e-3));  

  // string mkdirp = "mkdir -p " ;
  // Create_Dir(folderName);

  // poFF = new double[NFF];
  // GenFFPOs();
  
  nPostNeurons = new unsigned int[N_NEURONS];
  idxVec = new unsigned int[N_NEURONS];

  nPostNeuronsFF = new unsigned int[NFF];
  idxVecFF = new unsigned int[NFF];

  if(IF_REWIRE == 0) {
    if((trialNumber == 0 && phi_ext == 0) || IF_GEN_MAT) {
      clock_t timeStartCM = clock();
      if(IF_FIXED_FF_K) {
	GenFixedFFConMat();
      }
      else {
	GenFFConMat();
      }
      GenConMat(1); // the argument was used for testing kappa, setting it to 1 will generate a matrix with recMod = p
      delete [] conMat;
      clock_t timeStopCM = clock();
      double elapsedTimeCM = (double)(timeStopCM - timeStartCM) / CLOCKS_PER_SEC;  
      cout << "\n connection gen, elapsed time= " << elapsedTimeCM << "s, or " << elapsedTimeCM / 60.0 << "min" << endl;
    }
    else {
      printf("loading FF Sparse matrix\n");      
      LoadFFSparseConMat();
      printf("loading Sparse matrix\n");  
      LoadSparseConMat();
      // GenConMat(0);
      unsigned long nEE_IEConnections = 0;
      for(unsigned i = 0; i < N_NEURONS; i++) {
	nEE_IEConnections += nPostNeurons[i];
      }
      IS_REWIRED_LINK = new unsigned int[nEE_IEConnections];
      IF_LOADREWIREDCON = 0;
      if(IF_LOADREWIREDCON) { // = IF_REWIRE
	LoadRewiredCon(nEE_IEConnections);
      }
      else {
	for(unsigned long i = 0; i < nEE_IEConnections; i++) {
	  IS_REWIRED_LINK[i] = 0;
	}
      }
    }
  }
  else {
    printf("\n----------------------------------------\n");
    printf("---------------REWIRING-----------------\n");          
    printf("----------------------------------------\n");      
    printf("loading FF Sparse matrix\n");      
    LoadFFSparseConMat();
    printf("loading Sparse matrix\n");  
    LoadSparseConMat();
    unsigned long nEE_IEConnections = 0;
    for(unsigned i = 0; i < N_NEURONS; i++) {
      nEE_IEConnections += nPostNeurons[i];
    }

    IS_REWIRED_LINK = new unsigned int[nEE_IEConnections];    
    if(IF_LOADREWIREDCON && phi_ext > 0) {
      LoadRewiredCon(nEE_IEConnections);
    }
    else {
      for(unsigned long i = 0; i < nEE_IEConnections; i++) {
	IS_REWIRED_LINK[i] = 0;
      }
    }    

    /////////////////////////
    // if(sparseConVec != NULL) {
    //   printf("\ndeleting old sparseConVec\n");
    //   delete [] sparseConVec;
    //   sparseConVec = NULL;
    // }
    // GenConMat(1);
    //////////////////////////

    if(phi_ext == 0) { // REWIRE ONLY ONCE FOR ALL ANGLES
      unsigned int *conMatCONSTRUCTED = new unsigned int [(unsigned long int)N_NEURONS * N_NEURONS];
      printf("\n reconstructing matrix... ");  fflush(stdout);     
      GetFullMat(conMatCONSTRUCTED, sparseConVec, idxVec, nPostNeurons);
      printf("done\n");
      printf("\n removing connections... ");  fflush(stdout);  
      RemoveConnections(conMatCONSTRUCTED, kappa);
      printf("done\n");
      if(sparseConVec != NULL) {
      	printf("\ndeleting old sparseConVec\n");
      	delete [] sparseConVec;
      	sparseConVec = NULL;
      }
      if(IS_REWIRED_LINK != NULL) {
      	printf("\ndeleting old sparseConVec\n");
      	delete [] IS_REWIRED_LINK;
      	IS_REWIRED_LINK = NULL;
      }      
      printf("\n adding connections... ");  fflush(stdout);    
      AddConnections(conMatCONSTRUCTED, kappa);  
      printf("done\n");
      delete [] conMat;
      delete [] conMatCONSTRUCTED;
    }

    // unsigned int wrngcntrEE = 0, wrngcntrIE = 0, wrngcntrEI = 0, wrngcntrII = 0;
    // for (unsigned long int i = 0; i < N_NEURONS; i++)  {
    //   for (unsigned long int j = 0; j < N_NEURONS; j++)  {
    // 	if(conMat[i + N_NEURONS * j] != conMatCONSTRUCTED[i + N_NEURONS * j]) {
    // 	  if(i < NE && j < NE)       { wrngcntrEE += 1;}
    // 	  else if(i < NE && j >= NE) { wrngcntrIE += 1;}
    // 	  else if(i >= NE && j < NE) { wrngcntrEI += 1;}
    // 	  else                       { wrngcntrII += 1;}	  	  
    // 	  //	  else (i >= NE && j >= NE) { wrngcntrII += 1;}	  
    // 	}
    //   }
    // }
    // printf("oh la la! EE IE EI II= %u %u %u %u\n", wrngcntrEE, wrngcntrIE, wrngcntrEI, wrngcntrII);
  }
  
  //  exit(1);
  
  clock_t timeStart = clock(); 
  RunSim();
  clock_t timeStop = clock();
  double elapsedTime = (double)(timeStop - timeStart) / CLOCKS_PER_SEC;
  cout << "\n elapsed time = " << elapsedTime << "s, or " << elapsedTime / 60.0 << "min" << endl; 
  FILE *fpet = fopen("elapsedTime.txt", "a");
  fprintf(fpet, "%llu %f\n", nSteps, elapsedTime);
  fclose(fpet);

  // delete [] conMat;
  delete [] nPostNeurons;
  delete [] idxVec;
  delete [] sparseConVec;
  delete [] nPostNeuronsFF;
  delete [] idxVecFF;
  delete [] sparseConVecFF;
  delete [] IS_REWIRED_LINK;
  return 0; //EXIT_SUCCESS;
}
