function out = POofPop(tc)

[nNeurons, nAngles] = size(tc);
out = nan(1, nNeurons);
thetas = linspace(0, 180, nAngles);
for i = 1:nNeurons
    out(i) = PO(tc(i, :), thetas);
end
end

function out = PO(firingRate, atTheta) 
    out = nan;
    thetas = atTheta * pi / 180;
    
    x = cos(2 * thetas) * firingRate';
    y = sin(2 * thetas) * firingRate';
    out = atan2(y, x) * 180 / pi;
    
    
%    out = atan(y/x) * 180 / pi;
%     if(out > 0)
%         out = out / 2.0;
%     else
%         out = out / 2.0 + 180;
%     end
        
    if x < 0 && y > 0
        out = out;
    end
    
    if x > 0 && y < 0
        out = (out + 360);
    end
    if x < 0 && y < 0;
        out = (out + 360);
    end
    
    out = out / 2;
end

