% Train the 2020 pattern-recognition network on the window features and turn each
% prediction into a grip command for arduino/prosthetic_hand (class k closes finger k).
% Requires MATLAB with the Deep Learning Toolbox (patternnet); not re-run in 2026.
targets = full(ind2vec(labels));
net = patternnet(10);
net.divideParam.trainRatio = 0.6; net.divideParam.valRatio = 0.2; net.divideParam.testRatio = 0.2;
[net, tr] = train(net, feats, targets);
pred = vec2ind(net(feats(:, tr.testInd)));
fprintf('held-out windows: %d, accuracy %.2f (same recordings, different windows)\n', ...
        numel(tr.testInd), mean(pred == labels(tr.testInd)));
for k = unique(pred)
  angles = zeros(1, 5); angles(k) = 180;              % close finger k, others open
  fprintf('class %d -> G %d %d %d %d %d\n', k, angles);
end
