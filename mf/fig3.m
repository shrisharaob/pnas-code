function varargout = fig3(varargin)


  % Jab = [1, -2.8;
  %        1, -1.5];


  %% param set 0 
   Jab = [1, -1.5;
          1, -1.145];
  cff = 0.1
  Ea = [2; 1.35] * 9 * cff;
  %%
  
  
  % Ea = [2; 1];
  % Ea  = [2;
  %        1.45]  * cff


  mExtZero = 0.075
  mExtOne = 0.03

  
    Pcritical(mExtZero, Jab, Ea, cff) 


  N = 10000;
  nDeltas = 100;
  nMoments = 10;
  nThetas = 36;

  kappa = 0
  mOneGuess = 0.08;
    
    for jII_beta = 1:0.5:4.
        GenOSIdistr_vs_beta(N, nMoments, nDeltas, nThetas, kappa, mExtZero, mExtOne, mOneGuess, Jab, Ea, cff, jII_beta);
    end
end


    
        
    
