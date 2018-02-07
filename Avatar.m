classdef Avatar
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        v % Vertices
        f % Faces
        k1 = NaN(1,3) % right wrist
        k2 = NaN(1,3) % right armpit
        k3 = NaN(1,3) % right hip
        k6 = NaN(1,3) % left hip
        k4 = NaN(1,3) % left armpit
        k5 = NaN(1,3) % left wrist
        k7 = NaN(1,3) % right foot min
        k8 = NaN(1,3) % left foot min
        k9 = NaN(1,3) % crotch
        k10 = NaN(1,3) %right waist
        k11 = NaN(1,3) %left waist
        
        l_ankle = NaN(1,3)
        r_ankle = NaN(1,3)
        lShoulder = NaN(1,3)
        rShoulder = NaN(1,3)
        armMaxR = NaN(1,3)
        armMaxL = NaN(1,3)
        collar
        
        chestCircumference
        waistCircumference
        hipCircumference % Circumference of the hip
        rThighGirth
        lThighGirth
        rCalfCircumference %Circumference of right calf
        lCalfCircumference %Circumference of left calf
        r_wristgirth
        l_wristgirth
        r_forearmgirth
        l_forearmgirth
        r_bicepgirth
        l_bicepgirth
        r_ankle_girth
        l_ankle_girth
        
        volume
        surfaceArea
        
        leftArmLength
        rightArmLength
        collarScalpLength
        trunkLength
        lLegLength
        rLegLength
        crotchHeight
        
        bodyType
        
        lcurve
        rcurve
        fcurve
        bcurve
        
        lArmIdx % Indices of left arm
        rArmIdx % Indices of right arm
        legIdx % Indices of legs
        headIdx % Indices of head
        trunkIdx % Indices of trunk
         
        headSA
        trunkSA
        lArmSA
        rArmSA
        lLegSA
        rLegSA
        
        time
        
    end
    
    methods
               %% Constructor
        function self = Avatar(input)
            if isstruct(input)
                self.v=input.v;
                self.f=uint32(input.f);
            elseif ischar(input) 
                myObj = readObj(input);
                self.v = myObj.v;
                self.f = uint32(myObj.f);
            end
            
            % Rotate if necessary
            distX = max(self.v(:,1)) - min(self.v(:,1)); % maximum distance in x direction
            distY = max(self.v(:,2)) - min(self.v(:,2)); % maximum distance in x direction
            if distY > distX
                [self.v(:,1), self.v(:,2)] = rotate_person(self.v(:,1), self.v(:,2), -pi/2);
            end
            
            % deleting problem vertices
            bdyEdges = getBoundaryEdges(self);
            runs = 0;
            while problems(bdyEdges)
                [self.v, self.f] = deleteProblems(self,bdyEdges);
                bdyEdges = getBoundaryEdges(self);
                runs = runs +1;
            end
            
            % fining holes
            if runs ~= 0
              list = findHoles(bdyEdges);
            end

            %slowest ones are:
            % 3.1 - armSearch(self, 'r');
            %       armSearch(self, 'l');
            % 1.6 - getWrist(self);
            % 1.5 - getAnkleGirth(self);
            % 1.1 - getArmpits(self); 
            % delete tic toc when optimized
                
            % feet mins
            t = [];
            tic
            [self.k8, self.k7] = getLegsMin(self);
            t = [t;toc];
            
            % crotch
            tic
            [self.k9] = getCrotch(self);
            t = [t;toc];
            
            % armpits
            tic
            [self.k2, self.k4] = getArmpits(self); 
            t = [t;toc];  
            
            % correct Crotch
            tic
            [self.k9] = adjustCrotch(self); 
            t = [t;toc];
            
            % arms
            tic
            [~, self.rArmIdx, ~] = armSearch(self, 'r');
            [~, self.lArmIdx, ~] = armSearch(self, 'l');
            t = [t;toc];
            
            % shoulders
            tic
            [self.lShoulder, self.rShoulder] = getShoulders(self);
            t = [t;toc];
            
            % collar
            tic
            [self.collar] = getCollar(self);
            t = [t;toc];
            
            % wrists
            tic
            [self.k1, self.k5, self.r_wristgirth, self.l_wristgirth] = getWrist(self);
            t = [t;toc];
            
            % hips
            tic
            [self.hipCircumference, hipPoints] = getHip(self);
            self.k3 = hipPoints(1,:);
            self.k6 = hipPoints(2,:);
            t = [t;toc];
            
            % waist
            tic
            [self.waistCircumference, waistPoints] = getWaist(self);
            self.k10 = waistPoints(1,:);
            self.k11 = waistPoints(2,:);
            t = [t;toc];
            

           
            % arm lengths
            tic
            [self.leftArmLength, self.rightArmLength, self.armMaxR, self.armMaxL] = getArmLength(self);
            t = [t;toc];
            
            % length from collar to scalp
            tic
            [self.collarScalpLength] = getCollarScalpLength(self);
            t = [t;toc];
            
            % length from crotch to collar
            tic
            [self.trunkLength] = getTrunkLength(self);
            t = [t;toc];
            
            % length of legs
            tic
            [self.lLegLength, self.rLegLength] = getLegLength(self);
            t = [t;toc];
            
            %leg indices
            tic
            [self.legIdx] = getLegs(self);
            t = [t;toc];
            
            %ankles
            tic
            [self.l_ankle,self.r_ankle, self.r_ankle_girth, self.l_ankle_girth] = getAnkleGirth(self);
            t = [t;toc];
            
            %Calves
            tic
            [self.lCalfCircumference, self.rCalfCircumference] = getCalf(self);
            t = [t;toc];
            
            % crotch height
            tic
            [self.crotchHeight] = getCrotchHeight(self);
            t = [t;toc];
            
            % volume and surface area
            tic
            [self.volume, self.surfaceArea] = findSurfaceAreaAndVolume(self);
            t = [t;toc];
            
       
            % head and trunk
            tic
            self.headIdx = getHead(self);
            self.trunkIdx = getTrunk(self);
            t = [t;toc];
            
             % left and right thigh girths
            tic
            [self.rThighGirth, self.lThighGirth] = getThighGirth(self);
            t = [t;toc];
            
             %arm girths
            tic
            [self.r_forearmgirth, self.l_forearmgirth, self.r_bicepgirth, self.l_bicepgirth] = getArmGirth(self);
            t = [t;toc];
            
            % chest circumference
            tic
            self.chestCircumference = getChestCircumference(self);
            t = [t;toc];
            
            % front, back, left, and right curve
            tic
            [self.rcurve, self.lcurve, ~] = getCurve(self,1,12);
            [self.fcurve, self.bcurve, ~] = getCurve(self,2,12);
            t = [t;toc];
            self.time = t;
            
            % get body type
            tic
            self.bodyType = getBodyType(self);
            t = [t;toc];
            
            %surface area by part
           
            self.headSA = getComponentArea(self,self.headIdx);
            self.trunkSA = getComponentArea(self,self.trunkIdx);
            self.lArmSA = getComponentArea(self,self.lArmIdx);
            self.rArmSA = getComponentArea(self,self.rArmIdx);
            Xpts = self.v(:,1);
            Ypts = self.v(:,2);
            Zpts = self.v(:,3);
            lLeg = [Xpts(Xpts > 0 & Zpts < self.k9(3)),...
                    Ypts(Xpts > 0 & Zpts < self.k9(3)),...
                    Zpts(Xpts > 0 & Zpts < self.k9(3))];
            [~,lLegIdx,~] = intersect(lLeg,self.v);
            self.lLegSA = getComponentArea(self,lLegIdx);
            rLeg = [Xpts(Xpts < 0 & Zpts < self.k9(3)),...
                Ypts(Xpts < 0 & Zpts < self.k9(3)),...
                Zpts(Xpts < 0 & Zpts < self.k9(3))]; 
            [~,rLegIdx,~] = intersect(rLeg,self.v);
            self.rLegSA = getComponentArea(self,rLegIdx);
         end
  
        %% Methods
        
        function [var_v, var_f] = peelSkin(self,pCases)
            V = self.v;
            F = self.f;
            
            %figure; plot3(self.v(pCases,1),self.v(pCases,2),self.v(pCases,3),'-*'); hold on;
            for p = pCases'
                index = F(:,1) == p | F(:,2) == p | F(:,3) == p;
                F(index,:) = [];

                for j = 1:3
                   idx = F(:,j) > p; 
                   F(idx,j) = F(idx, j) - 1;
                end

                V(p,:) = [];
            end
            var_v = V;
            var_f = F;
        end
        
        function bdyEdges = getBoundaryEdges(self)
            % Find all edges in mesh, note internal edges are repeated
            E = sort([self.f(:,1) self.f(:,2); self.f(:,2) self.f(:,3); self.f(:,3) self.f(:,1)]')';
            % determine uniqueness of edges
            [u,~,n] = unique(E,'rows');
            % determine counts for each unique edge
            counts = accumarray(n(:), 1);
            % extract edges that only occurred once
            bdyEdges = u(counts==1,:);
        end
        
        function [v,f] = deleteProblems(self,bdyEdges)
            %bdyEdges = getBoundaryEdges(self);
            subs = [bdyEdges(:,1);bdyEdges(:,2)];
            A = accumarray(subs,1);
            vertices = find(A ~= 2 & A~=0);
            %vertices = unique([bdyEdges(:,1);bdyEdges(:,2)]);
            [v,f] = peelSkin(self,vertices);
        end
        
        function [lLeg, rLeg] = getLegsMin(self)
        % Gets legs vertices    
            v1 = self.v(:,1);
            v2 = self.v(:,2);
            v3 = self.v(:,3);
            [m_V, m_I] = min(v3);          %Gives lowest point on avatar, which
            leg1 = [v1(m_I) v2(m_I) m_V];  %would be on one of the feet

            if leg1(1,1) > 0   %If Leg1 is the left leg
                OppositeSide =[v1(v1<0) v2(v1<0) v3(v1<0)]; %Gives right side
            else               %If Leg1 is the right leg
                OppositeSide =[v1(v1>0) v2(v1>0) v3(v1>0)]; %Gives left side
            end

            [~, I] = min(OppositeSide(:,3));        %Gives lowest point on other leg
            leg2 = [OppositeSide(I, 1) OppositeSide(I, 2) OppositeSide(I, 3)];

            if leg1(1,1) < leg2(1,1)
                lLeg = leg2;
                rLeg = leg1;
            else
                lLeg = leg1;
                rLeg = leg2;
            end
        end
        
        function [k9] = getCrotch(self)
        % Gets the crotch (only x and z coords)
            [k9_1, k9_3] = find_minmax(self.v(:,1), self.v(:,3), self.k7(1,1), self.k8(1,1), 101);
            k9_2 = getMissingY(self,k9_1,k9_3);
            k9 = [k9_1, k9_2, k9_3];
        end
        
        function [y] = getMissingY(self,x,z)
            roundv1 = round(self.v(:,1),4);
            roundv3 = round(self.v(:,3),4);
            y = self.v(roundv1 == round(x,4)...
                              & roundv3 == round(z,4),:);
            y = median(y(:,2)); % find average if more than one was found
        end
        
        function [k9_adj] = adjustCrotch(self)
            k9_adj = self.k9;
            N = 10;
            zPoints = linspace(self.k9(3), (min(self.k2(3),self.k4(3))+self.k9(3))/2, N);
%             figure;%
            mx_v_bot = zeros(2,N); % max points in y direction
            delta_v2 = zeros(1,N);
            delta_v1 = zeros(1,N);
            cnd_vector = ones(1,N);
            for i = 1:N
                vOnLine = getVOnLine(self, self.v, zPoints(i), (1:length(self.v)));
                v1 = vOnLine(:,1);
                v2 = vOnLine(:,2);
                
                k = convhull(v1,v2);
                v1_cnvh = v1(k);
                v2_cnvh = v2(k);
                mean_v2 = mean(v2);
                
                % Get v1 and v2 of bottom
                idx = v2 < mean_v2;
                v2_bot = v2(idx);
                v1_bot = v1(idx);
                
                % Get v1 and v2 of bottom convexhull
                idx = v2_cnvh < mean_v2;
                v2h_bot = v2_cnvh(idx);
                v1h_bot = v1_cnvh(idx);    
                
                % Get v1 and v2 of bottom and middle
                qtrL = mean(v1_bot) + ((max(v1_bot)-mean(v1_bot))./2);
                qtrS = mean(v1_bot) + ((min(v1_bot)-mean(v1_bot))./2);
                idx = logical((v1_bot > qtrS) .* (v1_bot < qtrL)); % Indices of vertices of middle
                v1_bot = v1_bot(idx);
                v2_bot = v2_bot(idx);
                
                [mx_v_bot(2,i), Idx_v2_bot] = max(v2_bot);
                mx_v_bot(1,i) = v1_bot(Idx_v2_bot);
                Idx_mx_cvh = (v1h_bot == mx_v_bot(1,i));
%                 subplot(2,5,i); plot(v1_bot,v2_bot,'.'); hold on; %
%                 plot(mx_v1_bot(i),mx_v2_bot(i),'*'); hold on; %
%                 plot(v1h_bot,v2h_bot,'.'); hold on; %
               
                if (i~=1)                
                    if (sum(Idx_mx_cvh)>0) % Check if max is part of convexhull
                        cnd_vector(i) = 0;
                    else
                        mid_v1 = (max(v1_bot)+min(v1_bot)) ./ 2;
                        
                        % Get v1 of min of bottom, convex hull, and right side in y direction 
                        r_Idx = v1h_bot > mid_v1;
                        [v2h_botR, IdxR] = min(v2h_bot(r_Idx));
                        v1h_botR = v1h_bot(r_Idx); 
                        v1h_botR = v1h_botR(IdxR);
                        
                        % Get v1 of min of bottom, convex hull, and left side in y direction
                        l_Idx = v1h_bot < mid_v1;
                        [v2h_botL, IdxL] = min(v2h_bot(l_Idx));
                        v1h_botL = v1h_bot(l_Idx); 
                        v1h_botL = v1h_botL(IdxL);  
                        
                        delta_v1(i) = v1h_botR - v1h_botL;
                        
                        mid_v2h = (v2h_botR + v2h_botL)./ 2;
                        mid_v1h = (v1h_botR + v1h_botL)./ 2;
                        [~,IdxMid_v1_bot] = min(abs(v1_bot-mid_v1h));
                        mid_v2_bot = v2_bot(IdxMid_v1_bot);
                        delta_v2(i) = mid_v2_bot - mid_v2h;      
%                         plot(v1_bot(IdxMid_v1_bot),mid_v2_bot,'*'); %%
%                         plot(mid_v1h,mid_v2h,'*'); %%
%                         plot(v1h_botL,v2h_botL,'*'); %%
%                         plot(v1h_botR,v2h_botR,'*'); hold off; %%

                    end
                end
            end
            
            [t1,~] = kmeans(delta_v1',2);
            [s1,~] = kmeans(delta_v2',2);  
            
            i = 1;
            cnd = cnd_vector(i);
            while (i<N && cnd)
                i = i + 1;
                cnd = cnd_vector(i) * (t1(i)~=t1(1)) * (s1(i)~=s1(1));
            end
            
            % we want i-2 because of transition stages
            if (i>2)
                i = i-2;
            else % i==2
            	i = i-1;
            end

            k9_adj(3) = zPoints(i);
%             k9_adj = [mx_v1_bot(i),mx_v2_bot(i),zPoints(i)];
        end
        
        function [k2, k4] = getArmpits(self)
        % Gets the armpits x and z coords
            v1 = self.v(:,1);
            v3 = self.v(:,3);
            
            r_side = [v1(v1(:,1)< self.k9(1)), v3(v1(:,1)< self.k9(1))]; %Finds right half of body
            l_side = [v1(v1(:,1)> self.k9(1)), v3(v1(:,1)> self.k9(1))]; %Finds left half of body

            test_k2 = NaN(3,3);
            test_k4 = NaN(3,3);
            alphas = [pi/8, pi/12, pi/24];
            for i = 1:3
                alpha = alphas(i);
                [new_rs_v1, new_rs_v3] = rotate_person(r_side(:,1), r_side(:,2), alpha); %rotates right side to apply find_minmax function 

                [new_ls_v1, new_ls_v3] = rotate_person(l_side(:,1), l_side(:,2), -alpha); %rotates left side to apply find_minmax function

                [m_V_Right, m_I_Right] = min(new_rs_v1);
                rh = [m_V_Right, new_rs_v3(m_I_Right)];  %gives rotated right hand

                [m_V_Left, m_I_Left] = max(new_ls_v1);
                lh = [m_V_Left, new_ls_v3(m_I_Left)];    %gives rotated left hand

                [n_m_x,~] = rotate_person(self.k7(1,1), self.k7(1,3), alpha);
                [n_M_x,~] = rotate_person(self.k8(1,1), self.k8(1,3), -alpha);

                [rotated_k2_1, rotated_k2_3] = find_minmax(new_rs_v1,new_rs_v3, rh(1), n_m_x, 111); %Finds right armpit rotated
                [rotated_k4_1, rotated_k4_3] = find_minmax(new_ls_v1,new_ls_v3, n_M_x, lh(1), 111); %Finds left armpit rotated

                [test_k2(i,1), test_k2(i,3)] = rotate_person(rotated_k2_1, rotated_k2_3, -alpha); %Actual right armpit
                [test_k4(i,1), test_k4(i,3)] = rotate_person(rotated_k4_1, rotated_k4_3, alpha); %Actual left armpit
            end
            
            % delete points that are not in the armpit area
                % right
            p = test_k2(:,[1,3]);
            IC = incenter(triangulation([1,2,3],p),1);
            dist = zeros(1,3);
            for i = 1:3
                dist(i) = norm(IC-p(i,:));
            end
            [maxDist,maxDistIdx] = max(dist);
            dist(maxDistIdx) = [];
            if mean(dist)*2<maxDist
                test_k2(maxDistIdx,:) = [];
            end
                % left
            p = test_k4(:,[1,3]);
            IC = incenter(triangulation([1,2,3],p),1);
            dist = zeros(1,3);
            for i = 1:3
                dist(i) = norm(IC-p(i,:));
            end
            [maxDist,maxDistIdx] = max(dist);
            dist(maxDistIdx) = [];
            if mean(dist)*2<maxDist
                test_k4(maxDistIdx,:) = [];
            end
            
            [~,max_k2Idx] = max(test_k2(:,3));
            [~,max_k4Idx] = max(test_k4(:,3));
            
            k2 = test_k2(max_k2Idx,:);
            k4 = test_k4(max_k4Idx,:);
        end
        
        function [chestCircumference] = getChestCircumference(self)
        %chest circumference
            zValue = median([self.k2(3), self.k4(3)]); 
            vOnLine = getVOnLine(self, self.v, zValue, self.trunkIdx);
            chestCircumference = getCircumference(vOnLine(:,1), vOnLine(:,2));
        end

        function [armVertices, armVerticesIdx, faces] = armSearch(self, side)
        % Finds the vertices, their indexes and the faces of arm
        % side = 'r' or 'l'
            armVertices = [];
            armVerticesIdx = [];
            faces = [];
            
            try
%                 v1 = self.v(:,1);
%                 v3 = self.v(:,3);
                if side == 'r'
                    [~,handIdx] = min(self.v(:,1));
%                     armpitX = self.k2(1);
%                     armpitZ = self.k2(3);
                else
                	[~,handIdx] = max(self.v(:,1));
%                     armpitX = self.k4(1);
%                     armpitZ = self.k4(3);
                end
%                 armpit_idx = find(logical((round(v1,4) == round(armpitX,4)).*(round(v3,4) == round(armpitZ,4))));
%                 [~,armpit_idx_y] = min(self.v(armpit_idx,2));
%                 faces = [find(self.f(:,1) == armpit_idx(armpit_idx_y,1)); ...
%                     find(self.f(:,2) == armpit_idx(armpit_idx_y,1)); ...
%                     find(self.f(:,3) == armpit_idx(armpit_idx_y,1))];
                faces = [find(self.f(:,1) == handIdx); ...
                         find(self.f(:,2) == handIdx); ...
                         find(self.f(:,3) == handIdx)];
                armVerticesIdx = [];
                newFaces = faces; % faces to search with
                
                while(~isempty(newFaces))
                    newVerticesIdx = self.f(newFaces,:);
                    newVerticesIdx = newVerticesIdx(:);
                    newVerticesIdx = unique(newVerticesIdx);
                    newVertices = self.v(newVerticesIdx,:);
                    if side == 'r'
%                         newVerticesIdx = newVerticesIdx(newVertices(:,1) < armpitX);
                        newVerticesIdx = newVerticesIdx(newVertices(:,1) < self.k2(1));
                    elseif side == 'l'
%                         newVerticesIdx = newVerticesIdx(newVertices(:,1) > armpitX);
                        newVerticesIdx = newVerticesIdx(newVertices(:,1) > self.k4(1));
                    else
                        return
                    end
                    armVerticesIdx = [armVerticesIdx; newVerticesIdx]; % save vertices indexes
                    
                    % delete duplicates
                    [newFaces,~] = find(ismember(self.f,newVerticesIdx));
                    newFaces = unique(newFaces);
                    [~, ia, ~] = intersect(newFaces, faces);
                    newFaces(ia) = [];
                    
                    faces = [faces; newFaces]; % add new faces to old faces
                end
                
                armVerticesIdx = unique(armVerticesIdx);
                armVertices = self.v(armVerticesIdx,:);
            catch
                disp('Error: armSearch has armpit_idx that is not a scalar');
            end
        end
        
        function [legIdx] = getLegs(self)
        % Get vertex indices of legs
            % right side
            r1 = [self.k9(1) self.k9(3)];
            h1 = [self.k3(1) self.k3(3)];
            slopeR = (r1(2)-h1(2)) ./ (r1(1)-h1(1));
            y_interceptR = r1(2) - (slopeR * r1(1));
            v3 = self.v(:,3);
            v1 = self.v(:,1);
            wholeArea = v3 - (slopeR * v1) - y_interceptR;
            negAreaR = uint32(find(wholeArea<=0));
            
            % left side
            h1 = [self.k6(1) self.k6(3)];
            slopeL = (r1(2)-h1(2)) ./ (r1(1)-h1(1));
            y_interceptL = r1(2) - (slopeL * r1(1));
            wholeArea = v3 - (slopeL * v1) - y_interceptL;
            negAreaL = uint32(find(wholeArea<=0));
            
            % combine left and right
            legIdx = unique([negAreaL; negAreaR]);
            
            % we don't want parts of the arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            [~, ia, ~] = intersect(legIdx, armIdx);
            legIdx(ia) = [];
        end
        
        function [rThighGirth, lThighGirth] = getThighGirth(self)
            
            % Right Leg
            v1 = self.v(:,1);
            
            rHip = self.k3; 
            Crotch = self.k9;
            rHipCrotchMid = [(Crotch(1,1)+rHip(1,1))/2 , (Crotch(1,3)+rHip(1,3))/2]; %midpoint between hip and crotch
            rFoot = self.k7;
            dist_rHipMin = sqrt(((rHipCrotchMid(1,1)-rFoot(1,1))^2)+((rHipCrotchMid(1,2)-rFoot(1,3))^2)); %distance from bottom of foot to midpoint between hip and crotch
            alpha_r = asin((abs(rHipCrotchMid(1,1)-rFoot(1,1)))/(dist_rHipMin)); %angle of rotation for tight leg
            
            rLegIdx = self.legIdx(v1(self.legIdx) < self.k9(1,1));
            lLegIdx = self.legIdx(v1(self.legIdx) >= self.k9(1,1));

            [rot_rLeg_x,rot_rLeg_z] = rotate_person(self.v(:,1),self.v(:,3),alpha_r);

            z_rThigh = 0.75*(dist_rHipMin)+min(rot_rLeg_z); %z-coordinate of thigh-cross section on right leg
            vOnLine = getVOnLine(self, [rot_rLeg_x, self.v(:,2), rot_rLeg_z], z_rThigh, rLegIdx);
            rThighGirth = getCircumference(vOnLine(:,1), vOnLine(:,2));

            % Left Leg
            lHip = self.k6; 
            lFoot = self.k8;
            Crotch = self.k9;
            lHipCrotchMid = [(Crotch(1,1)+lHip(1,1))/2 , (Crotch(1,3)+lHip(1,3))/2]; %midpoint between hip and crotch on left leg
            dist_lHipMin = sqrt(((lHipCrotchMid(1,1)-lFoot(1,1))^2)+((lHipCrotchMid(1,2)-lFoot(1,3))^2)); %distance between bottom of foot and midpoint between hip and crotch on left leg
            alpha_l = asin((abs(lHipCrotchMid(1,1)-lFoot(1,1)))/(dist_lHipMin)); %angle of rotation for left leg

            [rot_lLeg_x,rot_lLeg_z] = rotate_person(self.v(:,1),self.v(:,3),-alpha_l); %rotated x and z coordinates of left leg

            z_lThigh = 0.75*(dist_lHipMin)+min(rot_lLeg_z); %z-coordinate of thigh-cross section on left leg

            vOnLine = getVOnLine(self, [rot_lLeg_x, self.v(:,2), rot_lLeg_z], z_lThigh, lLegIdx);
            lThighGirth = getCircumference(vOnLine(:,1), vOnLine(:,2)); 
        end
        
        function [headIdx] = getHead(self)
        % Get vertex indices of head
            headIdx = [];
            
            try
            maxShoulderZ = max(self.rShoulder(3), self.lShoulder(3));
            headIdx = uint32(find(self.v(:,3) > maxShoulderZ));
            
            % we don't want parts of the arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            [~, ia, ~] = intersect(headIdx, armIdx);
            headIdx(ia) = [];
            catch
                disp('Error: headIdx not found');
            end
        end
        
        function [lShoulder, rShoulder] = getShoulders(self) 
            lArmVertices = self.v(self.lArmIdx,:);
            [~, lShoulderZIdx] = max(lArmVertices(:,3));
            lShoulder = lArmVertices(lShoulderZIdx,:);
            
            rArmVertices = self.v(self.rArmIdx,:);
            [~, rShoulderZIdx] = max(rArmVertices(:,3));
            rShoulder = rArmVertices(rShoulderZIdx,:);
        end
        
        function [leftArmLength, rightArmLength,armMaxR, armMaxL] = getArmLength(self)
            leftArmLength = 0;
            rightArmLength = 0;
            try
                armMaxR = [(self.k4(1)+self.rShoulder(1))/2,(self.k4(3)+self.rShoulder(3))/2];
                armMaxL = [(self.k2(1)+self.lShoulder(1))/2,(self.k2(3)+self.lShoulder(3))/2];
                
                wristR = self.k1;
                wristL = self.k5;
                
                rightArmLength = sqrt((armMaxR(1,1)-wristR(1,1))^2+(armMaxR(1,2)-wristR(1,3))^2);
                leftArmLength = sqrt((armMaxL(1,1)-wristL(1,1))^2+(armMaxL(1,2)-wristL(1,3))^2);
            catch
                disp('Error: arm lengths not possible');
            end
        end
        
        function [collar] = getCollar(self)
            collar = (self.lShoulder+self.rShoulder)/2;
        end
        
        function [collarScalpLength] = getCollarScalpLength(self)
            v1 = self.v(:,1);
            v3 = self.v(:,3);
            [HeadZ, Head_I] = max(v3);
            HeadX = v1(Head_I);
            collarScalpLength = sqrt((HeadX - self.collar(1,1))^2 + (HeadZ - self.collar(1,2))^2);
        end
        
        function [trunkLength] = getTrunkLength(self)
            trunkLength = sqrt((self.k9(1)-self.collar(1,1))^2 + (self.k9(3)-self.collar(1,2))^2);
        end
        
        function [lLegLength, rLegLength] = getLegLength(self)
            % WE NEED TO THINK ABOUT THIS
            % The z value for the hips are the same
            % So the length of the left and right leg are the same
%             LegMaxR = [(self.k9(1)+self.k3(1))/2 , (self.k9(3)+self.k3(3))/2];
%             LegMaxL = [(self.k9(1)+self.k6(1))/2 , (self.k9(3)+self.k6(3))/2];
%             rLegLength = LegMaxR(1,2) - min(self.v(:,3));
%             lLegLength = LegMaxL(1,2) - min(self.v(:,3));
            legMaxZ = (self.k9(3)+self.k3(3))/2;
            legMinZ = min(self.v(:,3));
            rLegLength = legMaxZ - legMinZ;
            lLegLength = rLegLength;
        end
        
        function [crotchHeight] = getCrotchHeight(self)
            crotchHeight = self.k9(3) - min(self.v(:,3));
        end
           
        function [trunkIdx] = getTrunk(self)
        % Get vertex indices of trunk
            trunkIdx = (uint32(1:size(self.v,1)))';
            nonTrunkIdx = [self.lArmIdx; self.rArmIdx; ...
                           self.legIdx; self.headIdx];
            trunkIdx(nonTrunkIdx) = [];
        end
        
        function [circumference, end_points] = getHip(self)
        % For crotch and armpit points, may have to consider cases where
     
            % get all vertices without arms
            armIdx = [self.lArmIdx; self.rArmIdx];
            keepIdx = 1:length(self.v);
            keepIdx(armIdx) = [];
            
            zStart = self.k9(3);
            zEnd = (self.k2(3)+self.k9(3))/2;
            
            % Slice once then find the max slice
            % then slice that slice again
            
            [x,y,z] = slice_n_dice(self, 3, 10, zStart,zEnd, 1, keepIdx);

            % Finds end_points
            [x_min, x_min_idx] = min(x); %mins give right side point, maxes give left side point
            [x_max, x_max_idx] = max(x);
            y_min = y(x_min_idx);
            y_max = y(x_max_idx);
            mnZ = mean(z);
            end_points = [x_min, y_min(1), mnZ; x_max, y_max(1), mnZ];
            
            % Finds circumference of hip
            circumference = getCircumference(x,y);
        end
        
        function [circumference, end_points] = getWaist(self)
            z_mid = mean([self.k2(3),self.k3(3)]); 
            armIdx = [self.lArmIdx; self.rArmIdx];
            noArmIdx = (1:length(self.v));
            noArmIdx(armIdx) = [];
            vOnLine = getVOnLine(self, self.v, z_mid, noArmIdx);
            circumference = getCircumference(vOnLine(:,1), vOnLine(:,2));

            % Finds end_points
            [x_min, x_min_idx] = min(vOnLine(:,1)); %mins give right side point, maxes give left side point
            [x_max, x_max_idx] = max(vOnLine(:,1));
            y_min = vOnLine(x_min_idx,2);
            y_max = vOnLine(x_max_idx,2);
            mnZ = mean(vOnLine(:,3));
            end_points = [x_min, y_min(1), mnZ; x_max, y_max(1), mnZ];            
        end        
        
        function bodyType = getBodyType(self)
            hips = self.hipCircumference;
            waist = self.waistCircumference;
            chest = self.chestCircumference;
            if (waist < hips && waist<chest)
                if (abs(hips-chest)<1)
                    bodyType = 'Straight/Hourglass';
                else
                    bodyType = 'Pear';
                end
            else
                if (waist > chest)
                    bodyType = 'Round';
                else
                    bodyType = 'Inverted Triangle';
                end
            end
        end
            
        function [volume,surface] = findSurfaceAreaAndVolume(self)
            v1 = self.v(self.f(:,1),:);
            v2 = self.v(self.f(:,2),:);
            v3 = self.v(self.f(:,3),:);
            volume = sum(SignedVolumeOfTriangle(v1,v2,v3));
            surface = sum(normAll(crossAll(v2-v1,v3-v1)))/2;
        end
        
        function [k1, k5,rWristGirth,lWristGirth] = getWrist(self)

            % finding hands
            [~,rHandIdx] = min(self.v(:,1));
            right_hand = self.v(rHandIdx,:); %Gives the point on the right hand that is furthest to the right
            [~,lHandIdx] = max(self.v(:,1));
            left_hand =  self.v(lHandIdx,:); %Gives the point on the left hand that is furthest to the left

            % Establish rotation angle for arms
            hypotenuse = norm([self.lShoulder(1),self.lShoulder(3)] - [left_hand(1),left_hand(3)]);
            adjacent = self.lShoulder(3) - left_hand(3);
            theta_l = acos(adjacent/hypotenuse);
            
            hypotenuse = norm([self.rShoulder(1),self.rShoulder(3)] - [right_hand(1),right_hand(3)]);
            adjacent = self.rShoulder(3) - right_hand(3);
            theta_r = acos(adjacent/hypotenuse);
              
            % right arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),theta_r);
            [y,~] = rotate_person(self.v(:,2),self.v(:,3),theta_r);
            zStart = min(z(self.rArmIdx));
            zEnd = (3*zStart+max(z(self.rArmIdx)))/4;
            zStart = (3*zStart+zEnd)/4;
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            circumference = zeros(1,n);
            for i = 1:n
                vOnLine = getVOnLine(self, [x,y,z], zValue(i), self.rArmIdx);
                circumference(i) = getCircumference(vOnLine(:,1),vOnLine(:,2));
            end
            
            [rWristGirth, idx] = min(circumference);
            vOnLine = getVOnLine(self, [x,y,z], zValue(idx), self.rArmIdx);
            k1 = [mean(vOnLine(:,1)), mean(vOnLine(:,2)), mean(vOnLine(:,3))];
            
            % left arm
            [x,z] = rotate_person(self.v(:,1),self.v(:,3),-theta_l);
            [y,~] = rotate_person(self.v(:,2),self.v(:,3),-theta_l);
            zStart = min(z(self.lArmIdx));
            zEnd = (3*zStart+max(z(self.lArmIdx)))/4;
            zStart = (3*zStart+zEnd)/4;
            
            n = 20;
            zValue = linspace(zStart,zEnd,n);
            circumference = zeros(1,n);
            for i = 1:n
                vOnLine = getVOnLine(self, [x,y,z], zValue(i), self.lArmIdx);
                circumference(i) = getCircumference(vOnLine(:,1),vOnLine(:,2));
            end
            
            [lWristGirth, idx] = min(circumference);
            vOnLine = getVOnLine(self, [x,y,z], zValue(idx), self.lArmIdx);
            k5 = [mean(vOnLine(:,1)), mean(vOnLine(:,2)), mean(vOnLine(:,3))];
            
        end
        
        function [distl,distr,distrl] = getCurve(self, d, n)
        % Gets side or front curve
            % d = 1 for side curve
            % d = 2 for front curve
            if d ~= 1 && d ~= 2
                disp('Error: The value for d needs to be 1 or 2')
                return
            end
            zPoints = linspace(self.k9(3), min(self.k2(3),self.k4(3)), n);
            
            distrl = zeros(n,1); %distance of pos - neg
            distr = zeros(n,1); %distance of pos to y=0
            distl = zeros(n,1); %distance of neg to y=0
            
            a = reshape(self.v(self.f(:,:),3), [], 3);% z values for faces
            
            
            for i = 1:n
                facesOnLine = (a(:,1) >= zPoints(i) & (a(:,2) <= zPoints(i) | a(:,3) <= zPoints(i))) |...
                                   (a(:,2) >= zPoints(i) & (a(:,1) <= zPoints(i) | a(:,3) <= zPoints(i))) |...
                                   (a(:,3) >= zPoints(i) & (a(:,1) <= zPoints(i) | a(:,2) <= zPoints(i)));
                vIdxOnLine = reshape(self.f(facesOnLine,:),[],1);
                vIdxOnLine = unique(vIdxOnLine);
                trunkPlusLegs = unique([self.trunkIdx; self.legIdx]);
                vIdxOnLine = intersect(trunkPlusLegs,vIdxOnLine); % only vertices in trunk (no arms)
                vOnLine = self.v(vIdxOnLine,:);
                distrl(i) = max(vOnLine(:,d)) - min(vOnLine(:,d)); %range of RHS-LHS
                distr(i) = max(vOnLine(:,d)); %RHS >0
                distl(i) = min(vOnLine(:,d)); %LHS <0
            end
           
        end
        
        function [vOnLine] = getVOnLine(self, v, zValue, keepIdx)
        % Finds the vertices at a given z value
            % v are all vertices (they need to be rotated if necessary)
            % zValue is the given z value
            % keepIdx are the indices that should be keept (needed for
            %         chopping of)
            
            n = length(zValue);
            vOnLine = cell(1,n);
            z = reshape(v(self.f(:,:),3), [], 3); % z values of all faces
            for i = 1:n
                facesOnLine = (z(:,1) >= zValue(i) & (z(:,2) <= zValue(i) | z(:,3) <= zValue(i))) |...
                                       (z(:,2) >= zValue(i) & (z(:,1) <= zValue(i) | z(:,3) <= zValue(i))) |...
                                       (z(:,3) >= zValue(i) & (z(:,1) <= zValue(i) | z(:,2) <= zValue(i)));  
                vIdxOnLine = reshape(self.f(facesOnLine,:),[],1);
                vIdxOnLine = unique(vIdxOnLine);
                vIdxOnLine = intersect(keepIdx,vIdxOnLine); % only vertices that are in keepIdx
                vOnLine(i) = {self.v(vIdxOnLine,:)};
            end
            if n == 1
                vOnLine = vOnLine{1};
            end
        end
          
        function [rightforearmgirth, leftforearmgirth, rightbicepgirth, leftbicepgirth] = getArmGirth(self)
            A = [0;self.k1(3) - self.armMaxR(2)];
            B = [self.k1(1)-self.armMaxR(1); self.k1(3)-self.armMaxL(2)];
            C = [0;self.k5(3) - self.armMaxL(2)];
            D = [self.k5(1)-self.armMaxL(1); self.k5(3)-self.armMaxL(2)];
            
            right_arm_theta = acos(dot(A,B)/(norm(A)*norm(B)));
            left_arm_theta = acos(dot(C,D)/(norm(C)*norm(D)));
            
            %right wrist girth
            [rotated_rarm_x, rotated_rarm_z] = rotate_person(self.v(:,1), self.v(:,3), right_arm_theta);
            [~, rotated_right_wrist_z] = rotate_person(self.k1(1),self.k1(3), right_arm_theta);
            newV = [rotated_rarm_x, self.v(:,2), rotated_rarm_z];
            vOnLine = self.getVOnLine(newV, rotated_right_wrist_z, self.rArmIdx);
            rightwristgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));

            %left wrist girth
            [rotated_larm_x, rotated_larm_z] = rotate_person(self.v(:,1), self.v(:,3), -left_arm_theta);
            [~, rotated_left_wrist_z] = rotate_person(self.k5(1),self.k5(3), -left_arm_theta);
            newV = [rotated_larm_x, self.v(:,2), rotated_larm_z];
            vOnLine = self.getVOnLine(newV, rotated_left_wrist_z, self.lArmIdx);
            leftwristgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));
            
            % Forearm
            rightforearm = [((3*self.k1(1))+self.armMaxR(1))/4, ((3*self.k1(3))+self.armMaxR(2))/4];
            [~, rotated_right_forearm_z] = rotate_person(rightforearm(1),rightforearm(2), right_arm_theta);
            newV = [rotated_rarm_x, self.v(:,2), rotated_rarm_z];
            vOnLine = self.getVOnLine(newV, rotated_right_forearm_z, self.rArmIdx);
            rightforearmgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));
            
            leftforearm = [(3*self.k5(1)+self.armMaxL(1))/4, (3*self.k5(3)+self.armMaxL(2))/4];
            [~, rotated_left_forearm_z] = rotate_person(leftforearm(1),leftforearm(2), -left_arm_theta);
            newV = [rotated_larm_x, self.v(:,2), rotated_larm_z];
            vOnLine = self.getVOnLine(newV, rotated_left_forearm_z, self.lArmIdx);
            leftforearmgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));
            
            %Bicep
            rightbicep = [(self.k1(1)+3*self.armMaxR(1))/4, (self.k1(3)+3*self.armMaxR(2))/4];
            [~, rotated_right_bicep_z] = rotate_person(rightbicep(1),rightbicep(2), right_arm_theta);
            newV = [rotated_rarm_x, self.v(:,2), rotated_rarm_z];
            vOnLine = self.getVOnLine(newV, rotated_right_bicep_z, self.rArmIdx);
            rightbicepgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));
            
            leftbicep = [(self.k5(1)+3*self.armMaxL(1))/4, (self.k5(3)+3*self.armMaxL(2))/4];
            [~, rotated_left_bicep_z] = rotate_person(leftbicep(1),leftbicep(2), -left_arm_theta);
            newV = [rotated_larm_x, self.v(:,2), rotated_larm_z];
            vOnLine = self.getVOnLine(newV, rotated_left_bicep_z, self.lArmIdx);
            leftbicepgirth = getCircumference(vOnLine(:,1), vOnLine(:,2));
        end
        
        function [lAnkle, rAnkle, lAnkleGirth, rAnkleGirth] = getAnkleGirth(self)
            n = 20;
            
            % right leg
            zStart = min(self.v(self.v(:,1)<0,3));
            zEnd = (3*zStart + self.k9(3))/4;
            zStart = (7*zStart + zEnd)/8;
            
            zValue = linspace(zStart,zEnd,n);
            circumference = zeros(1,n);
            rLegIdx = intersect(find(self.v(:,1)<self.k9(1)),self.legIdx);
            for i = 1:n
                vOnLine = getVOnLine(self, self.v, zValue(i), rLegIdx);
                circumference(i) = getCircumference(vOnLine(:,1),vOnLine(:,2));
            end
            
            [rAnkleGirth, idx] = min(circumference);
            vOnLine = getVOnLine(self, self.v, zValue(idx), rLegIdx);
            rAnkle = [mean(vOnLine(:,1)), mean(vOnLine(:,2)), mean(vOnLine(:,3))];
            
            % left leg
            zStart = min(self.v(self.v(:,1)>0,3));
            zEnd = (3*zStart + self.k9(3))/4;
            zStart = (7*zStart + zEnd)/8;
            
            zValue = linspace(zStart,zEnd,n);
            circumference = zeros(1,n);
            lLegIdx = intersect(find(self.v(:,1)>self.k9(1)),self.legIdx);
            for i = 1:n
                vOnLine = getVOnLine(self, self.v, zValue(i), lLegIdx);
                circumference(i) = getCircumference(vOnLine(:,1),vOnLine(:,2));
            end
            
            [lAnkleGirth, idx] = min(circumference);
            vOnLine = getVOnLine(self, self.v, zValue(idx), lLegIdx);
            lAnkle = [mean(vOnLine(:,1)), mean(vOnLine(:,2)), mean(vOnLine(:,3))];
        end
        
        function [L_Calf_Circumference, R_Calf_Circumference] = getCalf(self)

            LegVal = 0; %Left
            %Left Leg 
            zStart_L = self.l_ankle(3);  %z-cord
            zEnd_L   = self.k9(3);       %Crotch

            [L_Calf_Circumference, ~] = calfCircumference(self, zStart_L, zEnd_L, LegVal);

            LegVal = 1; %Right
            %Right Leg
            zStart_R = self.r_ankle(3);  
            zEnd_R   = self.k9(3);       %Crotch

            [R_Calf_Circumference, ~] = calfCircumference(self, zStart_R, zEnd_R, LegVal);
        end

        function [circumference ,end_points] = calfCircumference(self, zStart, zEnd, LegVal)
        % Slice once then find the max slice
        % then slice that slice again

            % divide the body in half along vertical axes
            Xpts = self.v(:,1);
            Ypts = self.v(:,2);
            Zpts = self.v(:,3);

            if(LegVal == 0) %If left leg
                idx = find(Xpts > 0);
            else            %Else right leg
                idx = find(Xpts < 0);
            end
 
            [x,y,z] = slice_n_dice(self, 5, 10, zStart,zEnd, 1, idx);
            % Finds end_points
            [x_min, x_min_idx] = min(x); %mins give right side point, maxes give left side point
            [x_max, x_max_idx] = max(x);
            y_min = min(y(x_min_idx));
            y_max = max(y(x_max_idx));
            mnZ = mean(z);
            end_points = [x_min, y_min(1), mnZ; x_max, y_max(1), mnZ];

            % Finds circumference 
            circumference = getCircumference(x,y);
        end
        
        function [x,y,z] = slice_n_dice(self, n1, n2, zStart,zEnd, d, keepIdx)
            for n = [n1 n2] 
                zPoints = linspace(zStart, zEnd, n+1);
                dist    = zeros(n, 1);

                for i = 1:n
                    points = getVOnLine(self,self.v,mean([zPoints(i),zPoints(i+1)]),keepIdx);
                    yPoints = points(:,2);
                    dist(i) = max(yPoints);
                end

                if n == n2
                    dist= sosmooth3(dist, 7);
                end

                j=1;
                cnd = 1;
                while(cnd)
                     j= j+1;
                     cnd1 = j<=n-1;
                     cnd2 =  dist(j-1) < dist(j);
                     cnd = logical(cnd1*cnd2);
                end

                zStart = zPoints(j-1);
                zEnd = zPoints(j);
            end
            vOnLine = getVOnLine(self,self.v,mean([zPoints(j-1),zPoints(j)]),keepIdx);
            x = vOnLine(:,1);
            y = vOnLine(:,2);
            z = vOnLine(:,3);
        end
           

        function plot2d(self,keyPoints)
            % Plots the avatar in 2D front view
                if ~exist('keyPoints','var')
                    keyPoints = true;
                end

                figure; plot(self.v(:, 1),self.v(:, 3),'.');hold on;
                if keyPoints == true
                    plot(self.k1(1), self.k1(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k2(1), self.k2(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k3(1), self.k3(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k4(1), self.k4(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k5(1), self.k5(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k6(1), self.k6(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k9(1), self.k9(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k10(1), self.k10(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.k11(1), self.k11(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.lShoulder(1), self.lShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.rShoulder(1), self.rShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                    plot(self.collar(1), self.collar(3),'g--o','LineWidth',4,'MarkerSize',2); hold on;
                end
            end

        function plot3d(self)
        % Plots the avatar in 3D
            figure;
            lArmVertices = self.v(self.lArmIdx,:);
            plot3(lArmVertices(:,1),lArmVertices(:,2),lArmVertices(:,3),'.'); hold on;
            rArmVertices = self.v(self.rArmIdx,:);
            plot3(rArmVertices(:,1),rArmVertices(:,2),rArmVertices(:,3),'.'); hold on;
            legVertices = self.v(self.legIdx,:);
            plot3(legVertices(:,1),legVertices(:,2),legVertices(:,3),'.'); hold on;
            headVertices = self.v(self.headIdx,:);
            plot3(headVertices(:,1),headVertices(:,2),headVertices(:,3),'.'); hold on;
            trunkVertices = self.v(self.trunkIdx,:);
            plot3(trunkVertices(:,1),trunkVertices(:,2),trunkVertices(:,3),'.');
        end
            
        function plot3d_2(self)
            % left arm
            lArmFaces = getFaces(self,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self,self.rArmIdx);
            

            % head
            headFaces = getFaces(self, self.headIdx);
            

            % legs
            legFaces = getFaces(self, self.legIdx);
            
            % trunk
            trunkFaces = getFaces(self, self.trunkIdx);
           

            figure;
            h_lArm = patch('vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y');
            h_rArm = patch('vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm');
            h_head = patch('vertices', self.v, 'faces', headFaces, 'FaceColor', 'c');
            h_legs = patch('vertices', self.v, 'faces', legFaces, 'FaceColor', 'g');
            h_trunk = patch('vertices', self.v, 'faces', trunkFaces, 'FaceColor', 'b');

            set(h_lArm,'LineStyle','none')
            set(h_rArm,'LineStyle','none')
            set(h_head,'LineStyle','none')
            set(h_legs,'LineStyle','none')
            set(h_trunk,'LineStyle','none')

            light('Position',[-50,-50,50],'Style','infinite');
            light('Position',[50,50,50],'Style','infinite');
            lighting phong;
         end
        
        function Faces = getFaces(self,index)
             Faces = ismember(self.f, index);
             Faces = Faces(:,1) | Faces(:,2) | Faces(:,3);
             Faces = self.f(Faces,:);
         end
         
        function plotCurve(self,d)
            if d == 1
                distneg = self.rcurve;
                distpos = self.lcurve;
                dist = distpos - distneg;
            elseif d == 2
                distneg = self.fcurve;
                distpos = self.bcurve;
                dist = distpos - distneg;
            end
            n = size(dist);
            
            figure; 
            subplot(1,5,[2,3]); 
            plot(self.v(:, d),self.v(:, 3),'.'); 
            title('2D Image of Subject');
            hold on
            
            subplot(1,5,1)
            plot(distneg, 1:n,'-d')
            title('Length Negative - Center')
            ylabel('Line Number')
            
            subplot(1,5,4)
            plot(distpos, 1:n,'-d')
            title('Length Center - Positive')
            
            subplot(1,5,5)
            plot(dist, 1:n,'-d')
            title('Length of Line')
        end
        
        function plotAll(self)
            plot2d(self);
            plot3d(self);
            plotCurve(self,1);
            plotCurve(self,2);
        end
        
        
        function plot2d_gui(self,axes)
            % Plots the avatar in 2D front view

                plot(axes,self.v(:, 1),self.v(:, 3),'.');
                hold(axes,'on')
                    plot(axes,self.k1(1), self.k1(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.k2(1), self.k2(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.k3(1), self.k3(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.k4(1), self.k4(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.k5(1), self.k5(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.k6(1), self.k6(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.k9(1), self.k9(3),'g--o','LineWidth',4,'MarkerSize',2);
                    plot(axes,self.k10(1), self.k10(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.k11(1), self.k11(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.lShoulder(1), self.lShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.rShoulder(1), self.rShoulder(3),'g--o','LineWidth',4,'MarkerSize',2); 
                    plot(axes,self.collar(1), self.collar(3),'g--o','LineWidth',4,'MarkerSize',2);
                hold off
            end
        
        
        function plot3d_gui(self,axes)
            % left arm
            lArmFaces = getFaces(self,self.lArmIdx);
            
            % right arm
            rArmFaces = getFaces(self,self.rArmIdx);
            

            % head
            headFaces = getFaces(self, self.headIdx);
            

            % legs
            legFaces = getFaces(self, self.legIdx);
            
            % trunk
            trunkFaces = getFaces(self, self.trunkIdx);
           

            
            h_lArm = patch(axes,'vertices', self.v, 'faces', lArmFaces, 'FaceColor', 'y');
            h_rArm = patch(axes,'vertices', self.v, 'faces', rArmFaces, 'FaceColor', 'm');
            h_head = patch(axes,'vertices', self.v, 'faces', headFaces, 'FaceColor', 'c');
            h_legs = patch(axes,'vertices', self.v, 'faces', legFaces, 'FaceColor', 'g');
            h_trunk = patch(axes,'vertices', self.v, 'faces', trunkFaces, 'FaceColor', 'b');

            set(h_lArm,'LineStyle','none')
            set(h_rArm,'LineStyle','none')
            set(h_head,'LineStyle','none')
            set(h_legs,'LineStyle','none')
            set(h_trunk,'LineStyle','none')

            light('Position',[-50,-50,50],'Style','infinite');
            light('Position',[50,50,50],'Style','infinite');
            lighting phong;
        end
         
        function plot3d_points_gui(self,axes)
        % Plots the avatar in 3D
            
            lArmVertices = self.v(self.lArmIdx,:);
            hold(axes,'on');
            plot3(axes,lArmVertices(:,1),lArmVertices(:,2),lArmVertices(:,3),'.'); 
            rArmVertices = self.v(self.rArmIdx,:);
            plot3(axes,rArmVertices(:,1),rArmVertices(:,2),rArmVertices(:,3),'.'); 
            legVertices = self.v(self.legIdx,:);
            plot3(axes,legVertices(:,1),legVertices(:,2),legVertices(:,3),'.'); 
            headVertices = self.v(self.headIdx,:);
            plot3(axes,headVertices(:,1),headVertices(:,2),headVertices(:,3),'.'); 
            trunkVertices = self.v(self.trunkIdx,:);
            plot3(axes,trunkVertices(:,1),trunkVertices(:,2),trunkVertices(:,3),'.');
            hold(axes,'off');
        end
        
        function plotCurve_gui(self,d,laxes,caxes,raxes,daxes)
            if d == 1
                distneg = self.rcurve;
                distpos = self.lcurve;
                dist = distpos - distneg;
            elseif d == 2
                distneg = self.fcurve;
                distpos = self.bcurve;
                dist = distpos - distneg;
            end
            n = size(dist);
            plot(caxes,self.v(:, d),self.v(:, 3),'.'); 
            
           
            plot(laxes,distneg, 1:n,'-d')
            
            
            
            plot(raxes,distpos, 1:n,'-d') 
            
            plot(daxes,dist, 1:n,'-d')
        end
        
    end
      
end
%% Calves

%% other functions
function [p1, p3] = find_minmax(v1, v3, left, right,size)
    x_pts = linspace(left,right,size);      %Partiton x-interval from m to M with 101 points (100 regions)
    mn_v3 = zeros(length(x_pts)-1,1);   %holds min z-coordinates for each partition
    corr_v1 = zeros(length(x_pts)-1,1); %holds corresponding x-coordinate for min z-coordinate 

    for i=1:length(x_pts)-1               %For each region          
        tmp1 = x_pts(i+1) > v1;           %Define x-subinterval
        tmp2 = v1 > x_pts(i);
        portion_v3 = v3(logical(tmp1.*tmp2));   %The z-coordinates that are available in each subinterval 
        if isempty(portion_v3)
            mn_v3(i) = -Inf;
        else
            [mn_v3(i), mn_v3_Idx]= min(portion_v3); %Find min z-coordinate in subinterval and index
            portion_v1 = v1(logical(tmp1.*tmp2));   %x-coordinates that correspond to z-coordinates on portion_v3
            corr_v1(i) = portion_v1(mn_v3_Idx);     %gives x_coordinate that corresponds to min z-coordinate
        end
    end
    [p3, mxIdx_mn_v3] = max(mn_v3);    %Finds z-coordinate and index of crotch
    p1 = corr_v1(mxIdx_mn_v3);     %Gives x-coordinate of crotch
end

function [new_v1, new_v3] = rotate_person(v1, v3, alpha)
    person2d = [v1 v3];
    R = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)];
    person_rotated = (R*person2d')';
    new_v1 = person_rotated(:,1);
    new_v3 = person_rotated(:,2);
end

function output = sosmooth3(x,N) %%N is odd
 h = ones(N,1);
 output = conv(h,x);
 a = [(1:N)';N.*ones(length(output)-(2*N),1);(N:-1:1)'];
 output = output./a;
 n = (N-1)/2;
 cut = output(2*n+1:end-(2*n));
 for i = 1:n 
     cut = [output(2*(n-i+1)-1);cut;output(2*(n-1))];
 end
 output = cut;
end

function v = crossAll(v1,v2)
    % Finds the cross product for all pairs
    % v(1,:) = cross(v1(1,:),v2(1,:))
    % ...
    v = NaN(length(v1),3);
    v(:,1) = v1(:,2).*v2(:,3)-v2(:,2).*v1(:,3);
    v(:,2) = v1(:,3).*v2(:,1)-v2(:,3).*v1(:,1);
    v(:,3) = v1(:,1).*v2(:,2)-v2(:,1).*v1(:,2);
end

function surface = AreaOfTriangle(v1,v2,v3)
    surface =norm(cross((v2-v1),(v3-v1)))/2;
end

function compSurfArea = getComponentArea(self,Idx)
faces = self.f(Idx,:);
compSurfArea = 0;
for i = 1:length(Idx);
compSurfArea = compSurfArea + AreaOfTriangle(self.v(faces(i,1),:),...
                                 self.v(faces(i,2),:),...
                                 self.v(faces(i,3),:));
end
end

function norms = normAll(M)
    % Finds the norm for all rows
    norms = sqrt(sum(M.^2,2));
end

function c = getCircumference(x,y)
    b = boundary(x,y,0);         
    t_boundary = [x(b),y(b)];
    c = sum(normAll(t_boundary(1:end-1,:)-t_boundary(2:end,:)));
    end

function [volume] = SignedVolumeOfTriangle(v1,v2,v3)
    v321 = v3(1,1) * v2(1,2) * v1(1,3);
    v231 = v2(1,1) * v3(1,2) * v1(1,3);
    v312 = v3(1,1) * v1(1,2) * v2(1,3);
    v132 = v1(1,1) * v3(1,2) * v2(1,3);
    v213 = v2(1,1) * v1(1,2) * v3(1,3);
    v123 = v1(1,1) * v2(1,2) * v3(1,3);
    volume = (1/6)*(-v321+v231+v312-v132-v213+v123);
end

function check = problems(bdyEdges)
    check = false;
    subs = [bdyEdges(:,1);bdyEdges(:,2)];
    A = accumarray(subs,1);
    vertices = find(A ~= 2 & A~=0);
    if ~isempty(vertices)
        check = true;
    end
end

function list = findHoles(bdyEdges)
%     pCases = [];
% 
%     vertices = unique([bdyEdges(:,1);bdyEdges(:,2)]);
%     for vertex = vertices'
%         if sum(sum(bdyEdges == vertex)) == d
%             pCases = [pCases;vertex];
%         end
%     end
    list = zeros(1,length(bdyEdges));
    holeNum = 1;
    row = 1;
    col = 1;
    list(row) = holeNum;
    start = row;
    while true
        col = mod(col,2)+1; % get other col
        [a,b] = find(bdyEdges == bdyEdges(row,col));
        if sum([a(1),b(1)] == [row,col]) == 2
            row = a(2);
            col = b(2);
        else
            row = a(1);
            col = b(1);
        end
        list(row) = holeNum;
        
        if sum(find(list == 0)) == 0, break; end
        
        if row == start;
            holeNum = holeNum + 1;
            row = find(list == 0, 1);
            start = row;
            col = 1;
        end
        
    end
end


