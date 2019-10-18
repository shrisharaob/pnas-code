function [out, varargout] = SolveBeta(kappa, theta, delta, mExtZero, mExtOne, ...
                         mE1Guess, Jab, Ea, cff, varargin)
        
    %%%%% PARAMETERS %%%%%
    %%    Jab = [1, -1.5;
    %%           1, -1];
    %Ea = [2;
    %          1];
    %%%%% GAUSS HERMITE POLYNOMIALS %%%%%
    [hermX, hw] = GaussHermite(60);
    hermX = hermX .* sqrt(2.0);
    hw = hw ./ sqrt(pi);
    %%%%% MF SOLUTIONS %%%%%
    mA0 = -1.0 .* inv(Jab) * Ea .* mExtZero;
    QfuncInv = @(z) (sqrt(2) .* erfcinv(2 .* z));
    alphaA = Alpha(Jab, Ea, mExtZero, cff);
    uA0_tmp = -1 * sqrt(alphaA) .* QfuncInv(mA0);
    out_tmp = Solver(kappa, mExtZero, mExtOne, mE1Guess, Jab, Ea, cff);
    uE0 = out_tmp(1);
    mE1 = out_tmp(2);
    uI0 = out_tmp(3);
    uA0 = [uE0; uI0];    
    
    % Make a starting guess at the solution
    if length(varargin) >= 1
        x0 = varargin{1};
    else
        x0 = mA0; % + 0.001 * rand(); uncomment to test for convergence to the same solution
    end
    history = [];
    algo1 = 'levenberg-marquardt';
    algo2 = 'trust-region';
    algo3 = 'trust-region-dogleg';
    options = optimoptions(@fsolve, 'Display','off', 'TolX', 10e-16, 'TolFun', ...
                     10e-16, 'OutPutFcn', @myoutput, ...
                           'FiniteDifferenceType', 'central', ...
                           'Algorithm', algo1);

    myfunc = @(y) Beta(y, Jab, Ea, kappa, mExtZero, mExtOne, uA0, ...
                       alphaA, mE1, theta, delta, hermX, hw, cff);    

    % Solve the system
    try
        [x,fval,exitflag] = fsolve(myfunc, x0, options);
        if exitflag < 0
            keyboard                    
            %out = nan                   %
        else
            out = x; % abs(x) ; % qA0
        end
    catch ME
        ME
        disp('solver dosent converge - try a different alogrithm!')
        x0 = mA0 * 1e-3;
        out = nan
    end
        
    tmp_betaA = DefBeta(Jab, Ea, mExtZero, mExtOne, out, 0, 0, cff);
    format long
    %    disp(['uE = ', num2str(uA0(1) + 1), 'uI,  = ', num2str(uA0(2) + 1)]);          %disp(['alphaE = ', num2str(alphaA(1)), 'alphaI,  = ', num2str(alphaA(2))]);    
    %disp(['betaE = ', num2str(tmp_betaA(1)), ', betaI = ', num2str(tmp_betaA(2))]);
    %    out = Jab.^2 * qA0;

    function stop = myoutput(x,optimvalues,state);
        stop = false;
        if isequal(state,'iter')
          history = [history,  x];
        end
    end    

    varargout{1} = history;

    %    GenFrRates(mExtZero, uE0, alphaA(1), tmp_betaA(1));
    %    GenFrRates(mExtZero, uI0, alphaA(2), tmp_betaA(2));    
end

function out = GenFrRates(m0, uE, alphaE, betaE)
    x = randn(1, 10000);
    z = (-uE - sqrt(betaE) .* x) / sqrt(alphaE - betaE);
    SQRT_OF_2 = sqrt(2);
    MyErfc = @(z) 0.5 * erfc(z / SQRT_OF_2);
    out = MyErfc(z);
    disp('mean of generated mx');
    mean(out)
end


