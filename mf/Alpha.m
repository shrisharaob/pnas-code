function alphaA = Alpha(Jab, Ea, mExtZero, cff)
    mA0 = -1.0 .* inv(Jab) * Ea .* mExtZero;
    alphaA = Jab.^2 * mA0 + Ea.^2 * mExtZero / cff;
end