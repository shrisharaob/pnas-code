function out = Beta(x, Jab, Ea, kappa, mExtZero, mExtOne, uA0, alphaA, ...
                    mE1, theta, delta, hermX, hw, cff)
   
    % x(1) = qE0(delta), x(2) = qI0(delta)
    %%%%%%% CONSTANTS %%%%%%%%
    SQRT_OF_2 = sqrt(2);
    ONE_OVER_SQRT_OF_2 = 1 / sqrt(2);
    ONE_OVER_PI = 1.0 / pi;
    QUAD_TOL = 1e-8;
    MyErfc = @(z) 0.5 .* erfc(z ./ SQRT_OF_2);

    %%%%% QUENCHED DISORDER %%%%%
    BetaAFunc_delta0 = @(qA0) DefBeta(Jab, Ea, mExtZero, mExtOne, ...
                                      qA0, 0, 0, cff);
    BetaAFunc_of_delta = @(qA0, delta)  DefBeta(Jab, Ea, mExtZero, ...
                                                mExtOne, qA0, 0, delta, cff);

    %%%%% partial VARIANCE %%%%%
    alphaA_tilde = alphaA - abs(BetaAFunc_delta0(x));

    %%%%% UAbar(theta, delta) %%%%%
    uEplus  = @(theta, delta) (uA0(1) + (ONE_OVER_SQRT_OF_2 * Ea(1) * mExtOne + kappa * Jab(1, 1) * mE1)* cos(2 .* (theta + delta)));
    uEminus = @(theta, delta) (uA0(1) + (ONE_OVER_SQRT_OF_2 * Ea(1) * mExtOne + kappa * Jab(1, 1) * mE1)* cos(2 .* (theta - delta)));

    uIplus  = @(theta, delta) (uA0(2) + ONE_OVER_SQRT_OF_2 * Ea(2) * mExtOne * cos(2 .* (theta + delta)) );
    uIminus = @(theta, delta) (uA0(2) + ONE_OVER_SQRT_OF_2 * Ea(2) * mExtOne * cos(2 .* (theta - delta)) );

    %%%%% CORRECTION TERMS AFTER INTEGRATING OVER z_i %%%%%
    betaE0_one = ONE_OVER_SQRT_OF_2 * Ea(1) * mExtOne;
    betaI0_one = ONE_OVER_SQRT_OF_2 * Ea(2) * mExtOne;
    if(mExtOne == 0)
        correctionEplus = 0; correctionIplus = 0;
        correctionEminus = 0; correctionIminus = 0;
    else
        cosTDplus = @(theta, delta) cos(2 .* (theta + delta));
        cosTDminus = @(theta, delta) cos(2 .* (theta - delta));
        cosTDSqrd_plus = @(theta, delta) (cos(2 .* (theta + delta)).^2);
        cosTDSqrd_minus = @(theta, delta) (cos(2 .* (theta - delta)).^2);
        
        AEplus  = sqrt(betaE0_one ./ (alphaA_tilde(1) + betaE0_one * cosTDSqrd_plus(theta, delta) ));
        AEminus = sqrt(betaE0_one ./ (alphaA_tilde(1) + betaE0_one * cosTDSqrd_minus(theta, delta) ));
        AIplus  = sqrt(betaI0_one ./ (alphaA_tilde(2) + betaI0_one * cosTDSqrd_plus(theta, delta) ));
        AIminus = sqrt(betaI0_one ./ (alphaA_tilde(2) + betaI0_one * cosTDSqrd_minus(theta, delta) ));
        
        BEplus  = cosTDplus(theta, delta) * exp(- 2 * uEplus(theta, delta).^2 ./ (4 * alphaA_tilde(1) + betaE0_one * cosTDSqrd_plus(theta, delta)));
        BEminus = cosTDminus(theta, delta) * exp(- 2 * uEminus(theta, delta).^2 ./ (4 * alphaA_tilde(1) + betaE0_one * cosTDSqrd_minus(theta, delta)));
        BIplus  = cosTDplus(theta, delta) * exp(- 2 * uIplus(theta, delta).^2 ./ (4 * alphaA_tilde(2) + betaI0_one * cosTDSqrd_plus(theta, delta)));
        BIminus = cosTDminus(theta, delta) * exp(- 2 * uIminus(theta, delta).^2 ./ (4 * alphaA_tilde(2) + betaI0_one * cosTDSqrd_minus(theta, delta)));

        

        CEplus  = 1 - MyErfc(-(uEplus(theta, delta) .* sqrt(betaE0_one) .* cosTDplus(theta, delta)) ./ (2 * alphaA_tilde(1)^2 + betaE0_one * alphaA_tilde(1) * cosTDSqrd_plus(theta, delta) ));
        CEminus = 1 - MyErfc(-(uEminus(theta, delta) .* sqrt(betaE0_one) .* cosTDminus(theta, delta)) ./ (2 * alphaA_tilde(1)^2 + betaE0_one * alphaA_tilde(1) * cosTDSqrd_minus(theta, delta) ));
        CIplus  = 1 - MyErfc(-(uIplus(theta, delta) .* sqrt(betaI0_one) .* cosTDplus(theta, delta)) ./ (2 * alphaA_tilde(2)^2 + betaI0_one .* alphaA_tilde(2) .* cosTDSqrd_plus(theta, delta) ));
        CIminus = 1 - MyErfc(-(uIminus(theta, delta) .* sqrt(betaI0_one) .* cosTDminus(theta, delta)) ./ (2 * alphaA_tilde(2)^2 + betaI0_one .* alphaA_tilde(2) .* cosTDSqrd_minus(theta, delta) ));        

        correctionEplus  = AEplus * BEplus * CEplus;
        correctionEminus = AEminus * BEminus * CEminus;
        correctionIplus  = AIplus * BIplus * CIplus;
        correctionIminus = AIminus * BIminus * CIminus;                
    end

    %%%%% INTEGRATE OVER X %%%%%
    tmpPlus = BetaAFunc_of_delta(x, delta) - [betaE0_one; betaI0_one];
    betaAplus = abs(tmpPlus);
    tmpMinus = BetaAFunc_of_delta(x, -delta) - [betaE0_one; betaI0_one];
    betaAminus = abs(tmpMinus);

    mEiPlus  = MyErfc(-(uEplus(theta, delta) + sign(tmpPlus(1)) * sqrt(betaAplus(1)) .* hermX  ) ./ sqrt(alphaA_tilde(1))) + correctionEplus;
    mEiminus = MyErfc(-(uEminus(theta, delta) + sign(tmpMinus(1)) * sqrt(betaAminus(1)) .* hermX) ./ sqrt(alphaA_tilde(1))) + correctionEminus;
    mIiPlus  = MyErfc(-(uIplus(theta, delta) + sign(tmpPlus(2)) * sqrt(betaAplus(2)) .* hermX  ) ./ sqrt(alphaA_tilde(2))) + correctionIplus;
    mIiminus = MyErfc(-(uIminus(theta, delta) + sign(tmpMinus(2)) * sqrt(betaAminus(2)) .* hermX) ./ sqrt(alphaA_tilde(2))) + correctionIminus;
    qE0_of_delta = hw' * (mEiPlus .* mEiminus);
    qI0_of_delta = hw' * (mIiPlus .* mIiminus);
    out = [x(1) - qE0_of_delta; x(2) - qI0_of_delta];
end



    
