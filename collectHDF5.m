% outname = ['CLframes_' 'ALL_'];
% files = dir('Z:/FlyTracker/*.hdf5');
outname = ['CLframes_' '16mic_'];
files = dir('Z:/FlyTracker/1901*.hdf5');
allFrames = nan(3,192,192,300);
allPoints = zeros(18,300);
ind = 0;
for ff=1:length(files)
    try
        flyPoints = h5read([files(ff).folder '/' files(ff).name],'/flyPoints');
        frames = h5read([files(ff).folder '/' files(ff).name],'/flyFrames');
        for i=1:300
%             disp(num2str(i))
            if sumall(frames(:,:,:,i)) == 0
                break
            end
            ind = ind + 1;
            allFrames(:,:,:,ind) = frames(:,:,:,i);
            allPoints(1:size(flyPoints,1),ind) = flyPoints(:,i);
        end
    catch
    end
end

%%
allPoints = allPoints(:,1:ind);
allFrames = allFrames(:,:,:,1:ind);
allPoints = permute(allPoints,[2,1]);
allFrames = permute(allFrames,[4,3,2,1]);
%%

for scale = [1]
    inFrames = zeros(size(allPoints,1),192*scale,192*scale,3);
    outFrames = zeros(size(allPoints,1),192*scale,192*scale);
    for flynum = 1:size(allPoints,1)
        featureMap = zeros(192,192);
        if allPoints(flynum,1) ~= 0
            featureMap(drawline(allPoints(flynum,1:2),allPoints(flynum,3:4),[192,192])) = 1;
            featureMap(drawline(allPoints(flynum,5:6),allPoints(flynum,3:4),[192,192])) = 1;
        end
        if allPoints(flynum,7) ~= 0
            featureMap(drawline(allPoints(flynum,7:8),allPoints(flynum,9:10),[192,192])) = 1;
            featureMap(drawline(allPoints(flynum,11:12),allPoints(flynum,9:10),[192,192])) = 1;
        end
        if allPoints(flynum,13) ~= 0
            featureMap(drawline(allPoints(flynum,13:14),allPoints(flynum,15:16),[192,192])) = 1;
            featureMap(drawline(allPoints(flynum,17:18),allPoints(flynum,15:16),[192,192])) = 1;
        end
        
        pMap = imgaussfilt(featureMap,5);
        img = squeeze(allFrames(flynum,:,:,:)/255);

        img = imresize(img, scale);
        pMap = imresize(pMap, scale);
        
        inFrames(flynum,:,:,:) = img;
        if max(pMap(:)) > 0
            pMap = pMap ./ max(pMap(:));
        end
        outFrames(flynum,:,:) = pMap';
    end
    
    save([outname num2str(scale) '.mat'],'inFrames','outFrames','-v7.3')
end