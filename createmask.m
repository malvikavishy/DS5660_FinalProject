% Path to the dataset folder
datasetFolder = 'C:\Users\malvi\OneDrive - Vanderbilt\DS5660_FinalProject-main\dataset'; % Replace with your dataset folder path
outputFolder = 'C:\Users\malvi\OneDrive - Vanderbilt\DS5660_FinalProject-main\output'; % Folder to save masks (optional)

% List all PNG files in the folder
imageFiles = dir(fullfile(datasetFolder, '*.png'));

% Initialize matrices for names and masks
numImages = length(imageFiles);
nameMatrix = cell(numImages, 1); % Store image names
MaskMatrix = cell(numImages, 1); % Store binary masks

% Loop through each image
for i = 1:numImages
    % Full file path
    imagePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    
    % Extract name from the filename before '.png'
    [~, name, ~] = fileparts(imageFiles(i).name);
    nameMatrix{i} = name; % Store the name in the matrix

    % Read the image and alpha channel
    [img, ~, alphaChannel] = imread(imagePath);
    img(img == 0) = 0;
    % Create binary mask from alpha channel
    binaryMask = alphaChannel > 0;

    % Store the mask in the mask matrix
    MaskMatrix{i} = binaryMask;
    mask_images(:,:,i) = binaryMask;
    names(:,i) = cellstr(name);
    training_ip(:,:,i) = binaryMask;
    training_target(:,:,:,i) = img;
end

% Display the result for verification
disp('Extracted Names:');
disp(nameMatrix);
disp('Masks generated and stored in MaskMatrix.');

%%
% Initialize matrices to store augmented data for each transformation
rotate_aug = [];
flip_aug = [];
shift_aug = [];
scale_aug = [];
all_augmented_names = {};
all_augmented_targets = [];

for i = 1:numImages
    originalMask = mask_images(:,:,i);
    % Extract name from the filename before '.png'
    [~, name, ~] = fileparts(imageFiles(i).name);
    nameMatrix{i} = name; % Store the name in the matrix
    
    % Generate augmented samples
    for j = 1:25 % 25 augmentations per image
        % Initialize augmented image as original
        augMask = originalMask;
        
        % --- Random rotation ---
        angle = randi([-180, 180], 1); 
        augMask_rot = imrotate(augMask, angle, 'crop');  % Rotate image
        rotate_aug(:,:,i,j) = augMask_rot;  % Store rotated image
        
        % --- Horizontal flip ---
        if rand > 0.5
            augMask_flip = flip(augMask, 2);  % Flip horizontally
        else
            augMask_flip = augMask;  % No flip
        end
        flip_aug(:,:,i,j) = augMask_flip;  % Store flipped image
        
        % --- Vertical flip ---
        if rand > 0.5
            augMask_vert_flip = flip(augMask, 1);  % Flip vertically
        else
            augMask_vert_flip = augMask;  % No flip
        end
        flip_aug(:,:,i,j) = augMask_vert_flip;  % Store vertically flipped image
        
        % --- Random shift (translation) ---
        maxShift = 20; % Maximum shift in pixels
        shiftX = randi([-maxShift, maxShift]);
        shiftY = randi([-maxShift, maxShift]);
        augMask_shift = imtranslate(augMask, [shiftX, shiftY], 'FillValues', 255); % Fill with white background
        shift_aug(:,:,i,j) = augMask_shift;  % Store shifted image
        
        % --- Random scaling (zoom) ---
        scaleFactor = 0.8 + (rand * 0.4); % Random scale between 0.8 and 1.2
        augMask_scale = imresize(augMask, scaleFactor, 'Method', 'nearest'); % Resize image
        [rows, cols] = size(originalMask);
        augMask_scale = imresize(augMask_scale, [rows, cols], 'Method', 'nearest'); % Resize back to original dimensions
        scale_aug(:,:,i,j) = augMask_scale;  % Store scaled image
        
        % Store the names and target matrices for the augmentations
        all_augmented_names{end+1} = name; % Store name for each augmented image
        all_augmented_targets(:,:,:,end+1) = training_target(:,:,:,i); % Store target for each augmented image

    end
end

disp('Augmentation completed.');

