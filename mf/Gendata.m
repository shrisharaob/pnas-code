function varargout = Gendata()

  
  % Jab = [1, -1.5/2;
  %        1, -1/2];
  % Ea = [2; 1];
  % cff = 0.1

  %% param set 0 
   Jab = [1, -1.5;
          1, -1.145];
  cff = 1.0
  Ea = [2; 1.35] * 9 * 0.1;
  %%%%
  
  mExtZero = 0.075
  mExtOne = 0.03

  N = 1000000;
  nDeltas = 100;
  nMoments = 10;
  nThetas = 36;

  kappa = 0
  mOneGuess = 0.08;

  pc = Pcritical(mExtZero, Jab, Ea, cff)

  disp(['k_c = ', num2str(pc)])

  
  
  %%
  %  GenOSIdistr(N, nMoments, nDeltas, nThetas, kappa, mExtZero, mExtOne, mOneGuess, Jab, Ea, cff)
  %%
  %BifurcationCurve(mExtZero, 0, Jab, Ea, cff)
  % m1l = [0, 0.25, 0.5, 1] * mExtZero;
  % parfor (i=1:4, 4)
  %     m1 = m1l(i)
  %     BifurcationCurve(mExtZero, m1, Jab, Ea, cff)
  % end
  
  %% FIGure 5
  %  KappaVsMeanOSI(10000, nMoments, nDeltas, nThetas, 200, mExtZero, mExtOne, mOneGuess, Jab, Ea, cff); 
   
  %%


  
  
  kappaList = [9, 18]                   %

  keyboard
  % kappaList = [9, 17]
  
  for kappa = kappaList
      GenOSIdistr(N, nMoments, nDeltas, nThetas, kappa, mExtZero, mExtOne, mOneGuess, Jab, Ea, cff)
  end
  
end

