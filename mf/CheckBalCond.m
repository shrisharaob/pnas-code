function varargout = CheckBalCond(Jab, Ea)

    JEE = Jab(1, 1);
    JEI = Jab(1, 2);
    JIE = Jab(2, 1);
    JII = Jab(2, 2);
    %
    JE0 = Ea(1);
    JI0 = Ea(2);

    JE = -JEI/JEE;
    JI = -JII/JIE;
    E = JE0;
    I = JI0;

    if(abs(JEE * JII) > abs(JEI * JIE))
        error('NOT IN BALANCED REGIME!!!!!! ')
    end
    (JE < JI)
    (E/I < JE/JI)
    (E/I < 1)
    (JE < 1)    
    if((JE < JI) | (E/I < JE/JI) | (E/I < 1)) % | (JE < 1))
        error('NOT IN BALANCED REGIME!!!!!! ')
    else
        disp('params ok - in bal regime')
    end
end
