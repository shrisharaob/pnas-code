//#include <string>
//#include <cstring>

#define NE 10000
#define NI 10000
#define NFF 10000
#define K 1000
#define SQRT_K (sqrt((double)K))
#define THRESHOLD_E 1.0
#define THRESHOLD_I 1.0
#define JEE 1.0
#define JIE 1.0


#define TAU_FF 4.0
#define TAU_E 4.0
#define TAU_I 2.0


//--- parameters in v0
/* #define JEI -1.5 */
/* #define JII -1.0 */
/* #define JE0 2.0 */
/* #define JI0 1.0 */
/* #define JEI (-1.5 / JI_FACTOR) */
/* #define JII (-1.0 / JI_FACTOR) */


// new parameters

#define JI_FACTOR 1.0
#define JEI (-1.5 / JI_FACTOR)
#define JII (-1.145 / JI_FACTOR)  //(-1.0 / JI_FACTOR)
#define cFF 0.1
#define SQRT_KFF (sqrt((double)K * cFF))
#define JE0 1.8 
#define JI0 1.215
#define JE0_K (JE0 / (cFF * sqrt((double)K)))
#define JI0_K (JI0 / (cFF * sqrt((double)K)))
#define JEE_K (JEE / sqrt((double)K))
#define JEI_K (JEI / sqrt((double)K))
#define JIE_K (JIE / sqrt((double)K))
#define JII_K (JII / sqrt((double)K))
#define N_NEURONS (NE + NI)

#define T_STOP 4000.0

#define T_TRANSIENT (T_STOP * 0.1)
#define IF_GEN_MAT_DEFAULT 0
#define IF_SAVE_SPKS 0
#define IF_STEP_PHI0 0
#define IF_SAVE_INPUT 1

#define N_SEGMENTS 2

#define IF_REWIRE 0
#define IF_FIXED_FF_K 0

//unsigned int IF_REWIRE = IF_REWIRE_DEFAULT

/* unsigned IF_GEN_MAT = 0; */
double *poFF;
unsigned int IF_LOADREWIREDCON = IF_REWIRE;
double rewiredEEWeight = 1;
double m0, tStop, ffModulation;
double kappa;
double recModulationEE, recModulationEI, recModulationIE;
/* double *conMat, *conMatFF; */
unsigned int *conMat, *conMatFF;
int trialNumber;
unsigned int *nPostNeurons, *sparseConVec = NULL, *idxVec, *IS_REWIRED_LINK;
unsigned int *nPostNeuronsFF, *sparseConVecFF, *idxVecFF;

unsigned int IF_GEN_MAT = 0;
unsigned long long int nSteps;

double phi_ext, m0_ext, m1_ext;
