%Avatar Gui for Penningtion
function AVA_GUI(varargin)

fig = figure(...
    'Tag',             'fig',...
    'Name',            'AVA_GUI',...
    'NumberTitle',     'off',...
    'Visible',         'on',... 
    'MenuBar',         'none',...
    'Units',           'normalized',...
    'Position',        [.01 .1 .98 .85],...
    'UserData',        struct('Ava',[],'AvaName',{},'Switch_Control',[]));

%   ##################################################
%   #             Toolbar Generation                 #
%   ##################################################
  
  
% add toolbar
tb = uitoolbar('Parent',fig);

% get open icon png & convert to use in toolbar

path = strrep(mfilename('fullpath'),strcat('\',mfilename),'');
[img_open,~,alpha] = imread(fullfile(path,'AVA_GUI','file_open.png'));
openIcon = double(img_open)/256/256;
openIcon(~alpha) = NaN;
  
% get save icon png & convert to use in toolbar
[img_save,~,alpha] = imread(fullfile(path,'AVA_GUI','file_save.png'));
saveIcon = double(img_save)/256/256;
saveIcon(~alpha) = NaN;
  
% [openBtn] - open button
uipushtool(...
    'Parent',          tb,...
    'Tag',             'openBtn',...
    'CData',           openIcon,...
    'TooltipString',   'Open File',...
    'ClickedCallback', @Open_Button_Callback);
  
% [saveBtn] - save button
uipushtool(...
    'Parent',          tb,...
    'Tag',             'saveBtn',...
    'CData',           saveIcon,...
    'TooltipString',   'Save File',...
    'ClickedCallback', @Save_Button_Callback);

%   ##################################################
%   #                 Tab Generation                 #
%   ##################################################
   
% create tab group
tgroup = uitabgroup('Parent',fig);


tab1 = uitab(...
    'Parent',          tgroup,...
    'Tag',             'tab1',...
    'Title',           'General');
   
%Tab Two
tab2 = uitab(...
    'Parent',          tgroup,...
    'Tag',             'tab2',...
    'Title',           'Side Curve');

%Tab Three
tab3 = uitab(...
    'Parent',          tgroup,...
    'Tag',             'tab3',...
    'Title',           'Front Curve');


%   ##################################################
%   #                    Tab 1                       #
%   ##################################################


%%%%%%%%% Tables %%%%%%%%%
%------------------------------------------------
%[LengthPanel]
LengPan = uipanel(...
    'Parent',          tab1,...
    'Tag',             'LengPan',...
    'Title',           'Lengths',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .82 .34 .17]);

%[LengthHeaders]
Headers = {...
    '<html><center><font size=+1><b>Left Arm<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Arm<br />&nbsp;',...
    '<html><center><font size=+1><b>Collar-Scalp<br />&nbsp;',...
    '<html><center><font size=+1><b>Trunk<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Leg<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Leg<br />&nbsp;',...
    };

% [LengthTbl]
uitable(...
    'Parent',          LengPan,...
    'Tag',             'LengthTable',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank','bank','bank','bank','bank','bank'},...
    'ColumnEditable',  [false false false false false false],...
    'ColumnWidth',     {100 100 110 100 100 100},...
    'Units',           'normalized',...
    'Position',        [.01 .01 .985 .9],...
    'Data',            []);
%----------------------------------------------
%[CircPanel]
CircPan = uipanel(...
    'Parent',          tab1,...
    'Tag',             'CircPan',...
    'Title',           'Circumferences',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .65 .34 .17]);

%[CircHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Chest<br />&nbsp;',...
    '<html><center><font size=+1><b>Waist<br />&nbsp;',...
    '<html><center><font size=+1><b>Hip<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Calf<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Calf<br />&nbsp;',...
    };

% [CircTbl]
uitable(...
    'Parent',          CircPan,...
    'Tag',             'CircTable',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank','bank','bank','bank','bank'},...
    'ColumnEditable',  [false false false false false],...
    'ColumnWidth',     {120 120 120 120 120},...
    'Units',           'normalized',...
    'Position',        [.01 .01 .985 .9],...
    'Data',            []);
%-------------------------------------------------
%[GirthPanel]
GirthPan = uipanel(...
    'Parent',          tab1,...
    'Tag',             'GirthPan',...
    'Title',           'Girths',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .48 .34 .17]);

%[GirthHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Right Thigh<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Thigh<br />&nbsp;',...
    '<html><center><font size=+1><b>Righ Wrist<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Wrist<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Forearm<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Forearm<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Bicep<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Bicep<br />&nbsp;',...
    '<html><center><font size=+1><b>Right Ankle<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Ankle<br />&nbsp;',...
    };

% [GirthTbl]
uitable(...
    'Parent',          GirthPan,...
    'Tag',             'GirthTable',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank','bank','bank','bank','bank','bank',...
                        'bank','bank','bank','bank'},...
    'ColumnEditable',  [false false false false false false false...
                        false false false],...
    'ColumnWidth',     {120 120 120 120 120 120 120 120 120 120},...
    'Units',           'normalized',...
    'Position',        [.01 .01 .985 .9],...
    'Data',            []);
%-------------------------------------------------

%[BodPanel]
BodPan = uipanel(...
    'Parent',          tab1,...
    'Tag',             'BodPan',...
    'Title',           'Full Body Measurements',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .31 .34 .17]);

%[BodHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Surface Area<br />&nbsp;',...
    '<html><center><font size=+1><b>Volume<br />&nbsp;',...
    '<html><center><font size=+1><b>Body Type<br />&nbsp;'
    };

% [BodTbl]
uitable(...
    'Parent',          BodPan,...
    'Tag',             'BodTable',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank','bank','bank'},...
    'ColumnEditable',  [false false false],...
    'ColumnWidth',     {120 120 150},...
    'Units',           'normalized',...
    'Position',        [.01 .01 .985 .9],...
    'Data',            {});
%-------------------------------------------------
%[PartPanel]
PartPan = uipanel(...
    'Parent',          tab1,...
    'Tag',             'PartPan',...
    'Title',           'Volume & Surface Area by Body Part',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .11 .34 .2]);

%[PartHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Right Leg<br />&nbsp;',...
    '<html><center><font size=+1><b>Left leg<br />&nbsp;',...
    '<html><center><font size=+1><b>Righ Arm<br />&nbsp;',...
    '<html><center><font size=+1><b>Left Arm<br />&nbsp;',...
    '<html><center><font size=+1><b>Head<br />&nbsp;',...
    '<html><center><font size=+1><b>Torso<br />&nbsp;',...
    };

% [PartTbl]
uitable(...
    'Parent',          PartPan,...
    'Tag',             'PartTable',... 
    'RowName',         {'Surface Area','Volume'},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank','bank','bank','bank','bank','bank',...
                        'bank','bank','bank','bank'},...
    'ColumnEditable',  [false false false false false false false...
                        false false false],...
    'ColumnWidth',     {120 120 120 120 120 120 120 120 120 120},...
    'Units',           'normalized',...
    'Position',        [.01 .01 .985 .9],...
    'Data',            []);
%-------------------------------------------------
%%%%%%%% Axes %%%%%%%%%
threeDaxesPanel = uipanel(...
    'Parent',          tab1,...
    'Tag',             'AxesPan',...
    'Title',           '3d Plot',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.37 .11 .30 .865]);
twoDaxesPanel = uipanel(...
    'Parent',          tab1,...
    'Tag',             'AxesPan',...
    'Title',           '2d Plot',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.7 .11 .29 .865]);
axes(...
    'Parent',          threeDaxesPanel,...
    'Tag',             '3dgraph',...
    'Position',        [.05 .18 .9 .8]);
axes(...
    'Parent',          twoDaxesPanel,...
    'Tag',             '2dgraph',...
    'Position',        [.05 .18 .9 .8]);
%%%%%%%%% Buttons %%%%%%%%%
uicontrol(...
    'Parent',          threeDaxesPanel,...
    'Tag',             'ChangeButton',...
    'Style',           'pushbutton',...
    'String',          'Solid/Dotted',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.1 .05 .30 .05],...
    'Callback',         @switch_graph);

uicontrol(...
    'Parent',          threeDaxesPanel,...
    'Tag',             'Save3dButton',...
    'Style',           'pushbutton',...
    'String',          'Save Plot',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.58 .05 .30 .05],...
    'Callback',         @save_3d);

uicontrol(...
    'Parent',          twoDaxesPanel,...
    'Tag',             'Save2dButton',...
    'Style',           'pushbutton',...
    'String',          'Save Plot',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.1 .05 .80 .05],...
    'Callback',         @save_2d);

%   ##################################################
%   #                    Tab 2                       #
%   ##################################################

%[SidePanel]
SidePan = uipanel(...
    'Parent',          tab2,...
    'Tag',             'SidePan',...
    'Title',           'Side Curve',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .01 .98 .97]);
uicontrol(...
    'Parent',          SidePan,...
    'Tag',             'SLeftButton',...
    'Style',           'pushbutton',...
    'String',          'Save Left Side',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .8 .16 .05],...
    'Callback',         @save_left_side);

uicontrol(...
    'Parent',          SidePan,...
    'Tag',             'SDiffButton',...
    'Style',           'pushbutton',...
    'String',          'Save Right Side',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .7 .16 .05],...
    'Callback',         @save_right_side);

uicontrol(...
    'Parent',          SidePan,...
    'Tag',             'SDiffButton',...
    'Style',           'pushbutton',...
    'String',          'Save Curve Difference',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .9 .16 .05],...
    'Callback',         @save_diff_side);

uicontrol(...
    'Parent',          SidePan,...
    'Tag',             'S2dButton',...
    'Style',           'pushbutton',...
    'String',          'Save Subject Image',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .6 .16 .05],...
    'Callback',         @save_2d_side);

sDiffPan = uipanel(...
    'Parent',          SidePan,...
    'Tag',             'sDiffPan',...
    'Title',           'Curve Difference',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .01 .13 .98]);

twoDside = uipanel(...
    'Parent',          SidePan,...
    'Tag',             '2dSide',...
    'Title',           '2d Image of Subject',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.44 .01 .4 .98]);
leftSide = uipanel(...
    'Parent',          SidePan,...
    'Tag',             'LSide',...
    'Title',           'Left Side',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.30 .01 .13 .98]);
rightSide = uipanel(...
    'Parent',          SidePan,...
    'Tag',             'RSide',...
    'Title',           'Right Side',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.84 .01 .13 .98]);

%[BodHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Body Type<br />&nbsp;'
    };

% [BodType2]
uitable(...
    'Parent',          tab2,...
    'Tag',             'BodType2',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank'},...
    'ColumnEditable',  [false],...
    'ColumnWidth',     {120},...
    'Units',           'normalized',...
    'Position',        [.19 .43 .066 .1],...
    'Data',             {});
    
axes(...
    'Parent',          twoDside,...
    'Tag',             '2dSimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          rightSide,...
    'Tag',             'RSimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          leftSide,...
    'Tag',             'LSimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          sDiffPan,...
    'Tag',             'SDiffimage',...
    'Position',        [.05 .08 .9 .9]);

%   ##################################################
%   #                    Tab 3                       #
%   ##################################################


%[FrontPanel]
FrontPan = uipanel(...
    'Parent',          tab3,...
    'Tag',             'FrontPan',...
    'Title',           'Front Curve',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .01 .98 .97]);

uicontrol(...
    'Parent',          FrontPan,...
    'Tag',             'FFrontButton',...
    'Style',           'pushbutton',...
    'String',          'Save Front Side',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .8 .16 .05],...
    'Callback',         @save_left_front);

uicontrol(...
    'Parent',          FrontPan,...
    'Tag',             'FBackButton',...
    'Style',           'pushbutton',...
    'String',          'Save Back Side',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .7 .16 .05],...
    'Callback',         @save_right_front);

uicontrol(...
    'Parent',          FrontPan,...
    'Tag',             'FDiffButton',...
    'Style',           'pushbutton',...
    'String',          'Save Curve Difference',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .9 .16 .05],...
    'Callback',         @save_diff_front);


%[BodHeadrs]
Headers = {...
    '<html><center><font size=+1><b>Body Type<br />&nbsp;'
    };

% [BodType3]
uitable(...
    'Parent',          tab3,...
    'Tag',             'BodType3',... 
    'RowName',         {},...
    'ColumnName',      Headers,...
    'FontSize',        11,...
    'ColumnFormat',    {'bank'},...
    'ColumnEditable',  [false],...
    'ColumnWidth',     {120},...
    'Units',           'normalized',...
    'Position',        [.19 .43 .066 .1],...
    'Data',             {});

uicontrol(...
    'Parent',          FrontPan,...
    'Tag',             'F2dButton',...
    'Style',           'pushbutton',...
    'String',          'Save Subject Image',...
    'FontSize',        15,...
    'Units',           'normalized',...
    'Position',        [.14 .6 .16 .05],...
    'Callback',         @save_2d_front);
fDiffPan = uipanel(...
    'Parent',          FrontPan,...
    'Tag',             'sDiffPan',...
    'Title',           'Curve Difference',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.01 .01 .13 .98]);
twoDfront = uipanel(...
    'Parent',          FrontPan,...
    'Tag',             '2dFront',...
    'Title',           '2d Image of Subject',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.44 .01 .4 .98]);
leftFront = uipanel(...
    'Parent',          FrontPan,...
    'Tag',             'FSide',...
    'Title',           'Front Side',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.30 .01 .13 .98]);
rightFront = uipanel(...
    'Parent',          FrontPan,...
    'Tag',             'BSide',...
    'Title',           'Back Side',...
    'FontWeight',      'Bold',...
    'FontSize',        17,...
    'Units',           'normalized',...
    'Position',        [.84 .01 .13 .98]);

axes(...
    'Parent',          twoDfront,...
    'Tag',             '2dSimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          rightFront,...
    'Tag',             'Fimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          leftFront,...
    'Tag',             'Bimage',...
    'Position',        [.05 .08 .9 .9]);
axes(...
    'Parent',          fDiffPan,...
    'Tag',             'FDiffimage',...
    'Position',        [.05 .08 .9 .9]);

 %   ##################################################                           
 %   #                Callbacks                       #                           
 %   ################################################## 
 
% Toolbar Callbacks
%--------------------------------------------------------------------
function Open_Button_Callback(hObject,~)
handles = guihandles;
data = guidata(hObject);
[data.UserData.AvaName,fpath] = uigetfile(...
    {'*.obj','Waveform File(*.obj)'; ...
    },'Open Raw Data File');
try
if (fullfile(fpath,data.UserData.AvaName))%without this cancel on diaog box breaks gui
   h = msgbox('Opening Avatar...','Opening','modal');
   Ava = Avatar(fullfile(fpath,data.UserData.AvaName)); %takes some time
   data.UserData.Ava = Ava;
   wC = Ava.waistCircumference;
   hC =  Ava.hipCircumference;
   chC = Ava.chestCircumference;
   rCC = Ava.rCalfCircumference;
   lCC = Ava.lCalfCircumference;
   v = Ava.volume;
   sa =  Ava.surfaceArea;
   bt = Ava.bodyType;
   lAL =  Ava.leftArmLength;
   rAL = Ava.rightArmLength;
   csL = Ava.collarScalpLength;
   tL = Ava.trunkLength;
   lLL = Ava.lLegLength;
   rLL = Ava.rLegLength;
   cH = Ava.crotchHeight;
   rTG = Ava.rThighGirth;
   lTG = Ava.lThighGirth;
   rWG = Ava.r_wristgirth;
   lWG = Ava.l_wristgirth;
   rFG = Ava.r_forearmgirth;
   lFG = Ava.l_forearmgirth;
   rBG = Ava.r_bicepgirth;
   lBG = Ava.l_bicepgirth;
   rAG = Ava.r_ankle_girth;
   lAG = Ava.l_ankle_girth;
   hSA = Ava.headSA;
   tSA = Ava.trunkSA;
   laSA = Ava.lArmSA;
   raSA = Ava.rArmSA;
   llSA = Ava.lLegSA;
   rlSA = Ava.rLegSA;
   
   
   handles.BodType2.Data = {bt};
   handles.BodType3.Data = {bt};
   handles.LengthTable.Data = [lAL,rAL,csL,tL,lLL,rLL];
   handles.CircTable.Data = [chC,wC,hC,rCC,lCC];
   handles.GirthTable.Data = [rTG,lTG,rWG,lWG,rFG,lFG,rBG,lBG,rAG,lAG];
   handles.BodTable.Data = {sa,v,bt};
   handles.PartTable.Data = [rlSA,llSA,raSA,laSA,hSA,tSA;];
   twoDaxes = handles.AxesPan(1,1).Children(2,1);
   cla(twoDaxes);
   threeDaxes = handles.AxesPan(1,2).Children(length(handles.AxesPan(1,2).Children),1);
   cla(threeDaxes);
   Ava.plot2d_gui(twoDaxes);
   Ava.plot3d_points_gui(threeDaxes);
   rotate3d(threeDaxes,'on');
   data.UserData.Switch_Control = 0;
   leftside = handles.tab2.Children(2,1).Children(2,1).Children;
   rightside = handles.tab2.Children(2,1).Children(1,1).Children;
   centerside = handles.tab2.Children(2,1).Children(3,1).Children;
   difference = handles.tab2.Children(2,1).Children(4,1).Children;
   Ava.plotCurve_gui(1,leftside,centerside,rightside,difference);
   leftside = handles.tab3.Children(2,1).Children(2,1).Children;
   rightside = handles.tab3.Children(2,1).Children(1,1).Children;
   centerside = handles.tab3.Children(2,1).Children(3,1).Children;
   difference = handles.tab3.Children(2,1).Children(4,1).Children;
   Ava.plotCurve_gui(2,leftside,centerside,rightside,difference);

   
   guidata(hObject,data);
   if (exist('h','var'))
       delete(h);
   end
   uiwait(msgbox('Open Completed','Completed','modal'));
else
    disp('Open Canceled')
        
end
catch
    uiwait(msgbox('Error Loading File','Error','error','modal'));
end

function Save_Button_Callback(hObject,~)
warning('off','all');%disable all matlab warnings
data = guidata(hObject); %get data stored in gui
Name = {['Subject ' strrep(data.UserData.AvaName,'_Styku.obj','')]};
[rname,rpath] = uiputfile(... %get save-file name
    {'*.xls','Excel Workbook(*.xls)';'*.txt','Text File(*.txt)'},'Save Data');
if (rpath)
    
    disp('Saving...')
    h = msgbox('Saving Data...','Saving','modal');
    filename = fullfile(rpath,rname); %get filepath
   
   Ava = data.UserData.Ava;
   wC = Ava.waistCircumference;
   hC =  Ava.hipCircumference;
   chC = Ava.chestCircumference;
   rCC = Ava.rCalfCircumference;
   lCC = Ava.lCalfCircumference;
   v = Ava.volume;
   sa =  Ava.surfaceArea;
   lAL =  Ava.leftArmLength;
   rAL = Ava.rightArmLength;
   csL = Ava.collarScalpLength;
   tL = Ava.trunkLength;
   lLL = Ava.lLegLength;
   rLL = Ava.rLegLength;
   cH = Ava.crotchHeight;
   rTG = Ava.rThighGirth;
   lTG = Ava.lThighGirth;
   rWG = Ava.r_wristgirth;
   lWG = Ava.l_wristgirth;
   rFG = Ava.r_forearmgirth;
   lFG = Ava.l_forearmgirth;
   rBG = Ava.r_bicepgirth;
   lBG = Ava.l_bicepgirth;
   rAG = Ava.r_ankle_girth;
   lAG = Ava.l_ankle_girth;
   hSA = Ava.headSA;
   tSA = Ava.trunkSA;
   laSA = Ava.lArmSA;
   raSA = Ava.rArmSA;
   llSA = Ava.lLegSA;
   rlSA = Ava.rLegSA;
   
   circs =[wC,hC,chC,rCC,lCC]; 
   lngths = [lAL,rAL,csL,tL,lLL,rLL];
   grths = [rTG,lTG,rWG,lWG,rFG,lFG,rBG,lBG,rAG,lAG];
   misc = [v,sa,cH,hSA,tSA,laSA,raSA,llSA,rlSA];
   circs_titles = {'Waist Circumference','Hip Circumference','Chest Circumference',...
                   'Right Calf Circumference','Left Calf Circumference'};
   lngths_titles = {'Left Arm Length', 'Right Arm Length', 'Collar-Scalp Length',...
                    'Trunk Length','Left Leg Length', 'Right Leg Length'};
   grths_titles = {'Right Thigh Girth','Left Thigh Girth','Right Wrist Girth',...
                   'Left Wrist Girth','Right Forearm Girth','Left Forearm Girth',...
                   'Right Bicep Girth','Left Bicep Girth','Right Ankle Girth','Left Ankle Girth'};
   misc_titles = {'Total Volume','Total Surface Area','Chest Height','Head Surface Area','Torso Surface Area',...
                  'Left Arm Surface Area','Right Arm Surface Area','Left Leg Surface Area','Right Leg Surface Area'};
   if(strfind(rname,'.xls'))
    %Saving Numeric Data
    xlswrite(filename,Name,'Sheet1','A1');
    xlswrite(filename,circs_titles,'Sheet1','A3');
    xlswrite(filename,circs,'Sheet1','A4');
    xlswrite(filename,lngths_titles,'Sheet1','A7');
    xlswrite(filename,lngths,'Sheet1','A8');
    xlswrite(filename,grths_titles,'Sheet1','A11');
    xlswrite(filename,grths,'Sheet1','A12');
    xlswrite(filename,misc_titles,'Sheet1','A15');
    xlswrite(filename,misc,'Sheet1','A16');
    
   
    xlServer = actxserver('excel.application'); %Open Excel in COM Server
    xlWorkbooks = xlServer.Workbooks; %Get Workbook data
    xlFile = xlWorkbooks.Open(filename); %Open excel file
    xlSheetData = xlServer.ActiveWorkbook.Sheets; %get sheet data
    xlSheetData.Item(1).Range('A4:Z4').ColumnWidth = 20;

    xlFile.Save;
    xlFile.Close;
    xlServer.Quit;


    
    
   else
        dat = transpose(dat);
        save(rname,'dat','-ascii')
   end
    delete(h);
    uiwait(msgbox('Save Completed','Completed','modal'))
else
end
warning('on','all');%enable warnings

% Button Callbacks
%--------------------------------------------------------------------
function switch_graph(hObject,~)
data = guidata(hObject);
handles = guihandles(hObject);
threeDaxes = handles.AxesPan(1,2).Children(length(handles.AxesPan(1,2).Children),1);
if (data.UserData.Switch_Control)
cla(threeDaxes);
data.UserData.Ava.plot3d_points_gui(threeDaxes);
data.UserData.Switch_Control = 0;
else
cla(threeDaxes);
data.UserData.Ava.plot3d_gui(threeDaxes);
data.UserData.Switch_Control = 1;
end
guidata(hObject,data);

function save_3d(hObject,~)
handles = guihandles(hObject);
ax = handles.AxesPan(1,2).Children(length(handles.AxesPan(1,2).Children),1);
saveAxes(ax);

function save_2d(hObject,~)
handles = guihandles(hObject);
ax = handles.AxesPan(1,1).Children(2,1);
saveAxes(ax);

function save_diff_side(hObject,~)
handles = guihandles(hObject);
ax = handles.tab2.Children.Children(4,1).Children;
saveAxes(ax);

function save_2d_side(hObject,~)
handles = guihandles(hObject);
ax = handles.tab2.Children.Children(3,1).Children;
saveAxes(ax);

function save_left_side(hObject,~)
handles = guihandles(hObject);
ax = handles.tab2.Children.Children(2,1).Children;
saveAxes(ax);

function save_right_side(hObject,~)
handles = guihandles(hObject);
ax = handles.tab2.Children.Children(1,1).Children;
saveAxes(ax);
    
function save_diff_front(hObject,~)
handles = guihandles(hObject);
ax = handles.tab3.Children.Children(4,1).Children;
saveAxes(ax);

function save_2d_front(hObject,~)
handles = guihandles(hObject);
ax = handles.tab3.Children.Children(3,1).Children;
saveAxes(ax);

function save_left_front(hObject,~)
handles = guihandles(hObject);
ax = handles.tab3.Children.Children(2,1).Children;
saveAxes(ax);

function save_right_front(hObject,~)
handles = guihandles(hObject);
ax = handles.tab3.Children.Children(1,1).Children;
saveAxes(ax);

 function saveAxes(ax)
ax.Units = 'pixels';
pos = ax.Position;
marg = 40;
rect =  [-marg, -marg, pos(3)+2*marg, pos(4)+2*marg];
F = getframe(ax,rect);
Image = frame2im(F);
 [filename,filepath] = uiputfile(... %get excel save-file name
    {'*.jpg','JPG(*.jpg)';'*.png','PNG(*.png)';'*.tif','TIFF(*.tif)'},'Save Data');
if(filepath)
filepath = filepath(1:end-1);
tmp = fullfile(cd,filename);
imwrite(Image, filename);
ax.Units = 'normalized';
if ~(strcmp(cd,filepath))
 movefile(tmp,filepath,'f');
end
uiwait(msgbox('Save Completed','Completed','modal'))
end

 
