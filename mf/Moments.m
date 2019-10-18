function out = Moments(x, Jab, Ea, kappa, mExtZero, mExtOne, uA0, alphaA, mA0)
% x(1) = uE0, x(2) = mE1, x(3) = uI0, x(4) = mI1
% REMINDER ::: X(1) & X(3) ARE INPUT MOMENTS, UND X(3) & X(4) ARE OUTPUT MOMENTS    

    x(2) = abs(x(2));
    x(4) = abs(x(4));    
    %%%%%%% CONSTANTS %%%%%%%%
    SQRT_OF_2 = sqrt(2);
    ONE_OVER_SQRT_OF_2 = 1 / sqrt(2);
    ONE_OVER_PI = 1.0 / pi;
    QUAD_TOL = 1e-6;
    MyErfc = @(z) 0.5 * erfc(z / SQRT_OF_2);    
    %%%%% INPUT MOMENTS  %%%%%
    uA0 = [x(1), x(3)];
    uE1 = @(theta) ((ONE_OVER_SQRT_OF_2 * Ea(1) * mExtOne + kappa * Jab(1, 1) * x(2))* cos(2 * theta));
    uI1 = @(theta) (ONE_OVER_SQRT_OF_2 * Ea(2) * mExtOne * cos(2 * theta));
    uE = @(theta) uA0(1) + uE1(theta);
    uI = @(theta) uA0(2) + uI1(theta);
    %%%%% OUTPUT MOMENTS %%%%%
    funcE0 = @(theta) MyErfc(-uE(theta) ./ sqrt(alphaA(1)));
    funcE1 = @(theta) (MyErfc(-uE(theta) ./ sqrt(alphaA(1))) .* cos(2 * theta));
    funcI0 = @(theta) MyErfc(-uI(theta) ./ sqrt(alphaA(2)));
    funcI1 = @(theta) (MyErfc(-uI(theta) ./ sqrt(alphaA(2))) .* cos(2 * theta));

    out(1) = mA0(1) - ONE_OVER_PI * quad(funcE0, 0, pi, QUAD_TOL); 
    out(2) = x(2) - 2 * ONE_OVER_PI * abs(quad(funcE1, 0, pi, QUAD_TOL));
    out(3) = mA0(2) - ONE_OVER_PI * quad(funcI0, 0, pi, QUAD_TOL);
    out(4) = x(4) - 2 * ONE_OVER_PI * quad(funcI1, 0, pi, QUAD_TOL);
end

