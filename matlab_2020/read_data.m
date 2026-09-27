% Load the five recordings (one per finger movement, 1000 samples each) ONCE,
% and cut each into 10 non-overlapping windows of 100 samples.
% The 2020 version loaded every file 10 times and rectified only the first copy,
% so its 50 "samples" were only 10 distinct vectors.
n_windows = 10; win = 100;
X = zeros(win, 5 * n_windows);            % columns = windows
labels = zeros(1, 5 * n_windows);
for k = 1:5
  s = abs(csvread(fullfile('data', sprintf('%d.csv', k))));   % full-wave rectification
  s = s(1:n_windows * win);
  X(:, (k-1)*n_windows + (1:n_windows)) = reshape(s, win, n_windows);
  labels((k-1)*n_windows + (1:n_windows)) = k;
end
