function [ answ ] = FMD( binned_signal )
% Half of the total Welch power of each column (named FMD in the 2020 code;
% despite the old comment it is not a median frequency).

[R C] = size(binned_signal);
answ = zeros(1,C);
for i = 1:C
    answ(1,i) = (1/2)*sum(pwelch(binned_signal(:,i)));
end

end
