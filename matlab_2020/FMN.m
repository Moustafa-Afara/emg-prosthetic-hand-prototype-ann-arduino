function [ answ ] = FMN( binned_signal )
% Mean frequency of each column: the Welch-spectrum-weighted average
% of the normalised frequency axis.

[R C] = size(binned_signal);
answ = zeros(1,C);

for i = 1:C
    [Pxx, W] = pwelch(binned_signal(:,i));
    answ(1,i) = (sum(W.*Pxx))/(sum(Pxx));
end

end



