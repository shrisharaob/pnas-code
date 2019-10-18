function out = Miofx(x, Jab, Ea, kappa, mExtZero, mExtOne, uA0, ...
                     alphaA, mE1, qA0, theta, cff)

    % x is the correlated Gaussian noise 
    % order parameters: uA0, qA0, mE1
    %
    
    delta = 0;
    SQRT_OF_2 = sqrt(2);
    ONE_OVER_SQRT_OF_2 = 1 / sqrt(2);
    QFunction = @(z) 0.5 .* erfc(z ./ SQRT_OF_2);

    %%%%% QUENCHED DISORDER %%%%%
    % beta at Delta = 0
    BetaAFunc_delta0 = @(qA0) DefBeta(Jab, Ea, mExtZero, mExtOne, ...
                                      qA0, 0, delta, cff);

    %%%%% Temporal fluctuations %%%%%
    alphaA_tilde = alphaA - abs(BetaAFunc_delta0(qA0));

    %%%%% UAbar(theta, delta) %%%%%
    uE  = @(theta) (uA0(1) + (ONE_OVER_SQRT_OF_2 * Ea(1) * mExtOne + kappa * Jab(1, 1) * mE1)* cos(2 .* theta));
    uI  = @(theta) (uA0(2) + ONE_OVER_SQRT_OF_2 * Ea(2) * mExtOne * cos(2 .* theta));

    %%% Tuning curve %%%%
    
    % FF input 
    z = sqrt(-2 * log(rand(2, 1)));
    ffInput = repmat(Ea .* mExtOne .* z ./ sqrt(cff), 1, numel(theta)) .* ...
              repmat(cos(2 .* theta), 2, 1);


    denom = alphaA_tilde; % - Ea .^2 .* mExtOne ^2  ./ (2 * cff);

    mEi  = QFunction(-(uE(theta) + x(1, :) + ffInput(1, :)) ./ sqrt(denom(1))); 
    mIi  = QFunction(-(uI(theta) + x(2, :) + ffInput(1, :)) ./ sqrt(denom(2)));  
     
    out = [mEi; mIi];
end



    
