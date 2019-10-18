function out = Pcritical(mExtZero, Jab, Ea, cff)

    %%%%% PARAMETERS %%%%%
    % Jab = [1, -1.5;
    %        1, -1];
    % Ea = [2; 1];
    % cff = 0.1;
    %%%%%
    CheckBalCond(Jab, Ea)
    mA0 = -1.0 .* inv(Jab) * Ea .* mExtZero
    QfuncInv = @(z) (sqrt(2.0) .* erfcinv(2.0 .* z));
    %%
    alphaA = Alpha(Jab, Ea, mExtZero, cff) ; %Jab.^2 * mA0 + Ea.^2 * mExtZero;
    % alphaA = Jab.^2 * mA0;
    % alphaA = Jab.^2 * mA0 + Ea.^2 * mExtZero;
    %%
    uA0 = -1 * sqrt(alphaA) .* QfuncInv(mA0);
    uE0 = uA0(1);
    alpha = alphaA(1);
    JEE = Jab(1, 1);
    %%%%%
    b = uE0 / sqrt(alpha);
    C0 = exp(-b^2 / 2.0) / sqrt(2.0 * pi);
    num = sqrt(alpha);
    denom = C0 * JEE;
    out =  num / denom;
end
    