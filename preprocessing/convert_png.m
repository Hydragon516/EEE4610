clc; clear all;
close all;

%% data load 경로 설정
% dcm file 경로
dcm_path = 'C:\Users\user\Documents\lung\NSCLC-Radiomics';
png_path = 'C:\Users\user\Documents\lung\png';
folder_list = dir(fullfile(dcm_path));

%% img load and save 16bit png file
parfor i = 3 : length(folder_list)
    folder_select = folder_list(i);
    folder_path = [folder_select.folder filesep folder_select.name];
    folder_select.name
    [img_list, tumor_list] = read_dcm(folder_path);
    
    if length(tumor_list) > 1
        if isa(img_list, 'int16')
            img_list = img_list + 1024;
            img_list = uint16(img_list);
        end
        
        for x = 1: size(img_list,4)
            img_path = [png_path filesep 'image' filesep folder_select.name(7:end) '_' num2str(x,'%06d') '.png'];
            imwrite(img_list(:,:,:,x),img_path,'BitDepth', 16)
        end
        for y = 1: size(tumor_list,4)
            mask_path = [png_path filesep 'mask' filesep folder_select.name(7:end) '_' num2str(y,'%06d') '.png'];
            imwrite(tumor_list(:,:,:,y),mask_path,'BitDepth', 1)
        end
    end
end

%% read dcm file
function [img, tumor] = read_dcm(folder_path)
filepath1 = dir(fullfile(folder_path)); filepath1 = filepath1(3:end);

for i=1:2
    if strfind(filepath1(i).name,'CTLUNG')
        linkfolder_tumor = [folder_path filesep filepath1(i).name];
        link_path_tumor = dir(fullfile(linkfolder_tumor)); link_path_tumor = link_path_tumor(3:end);
        for j=1:2
            if strfind(link_path_tumor(j).name,'Segmentation')
                linkfolder = [link_path_tumor(j).folder filesep link_path_tumor(j).name];
                loadpath = dir(fullfile(linkfolder)); loadpath = loadpath(3:end);
                loadpath_tumor = [loadpath.folder filesep loadpath.name];
            end
        end
    else
        linkfolder = [folder_path filesep filepath1(i).name];
        linkpath_img = dir(fullfile(linkfolder)); linkpath_img = linkpath_img(3:end);
        loadpath_img = [linkpath_img.folder filesep linkpath_img.name];
    end
end

img = dicomreadVolume(loadpath_img); % 3차원 데이터
try
    tumor = dicomread(loadpath_tumor); % 3차원 데이터
catch
    tumor = [];
end

return
end