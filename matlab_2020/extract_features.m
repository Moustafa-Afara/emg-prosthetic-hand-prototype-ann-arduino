% Four spectral features per window (columns of X from read_data.m), each scaled to max 1.
feats = [FMD(X); FR(X); FMN(X); MFMD(X)];            % 4 x windows
feats = feats ./ max(feats, [], 2);
