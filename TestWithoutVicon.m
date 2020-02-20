clear, clc

Start = false;

Target = 250;

Tolerance =5;

%Array we will store Xcoordinates for experiment
Xcoordinate = [];

%Array for storing whether or not in target area
TargetArea = [];

%Construct an instance of Reaction
Reaction1 = Reaction('127.0.0.1',13000);
Reaction1.Connect();

% A dialog to stop the loop
MessageBox = msgbox( 'Stop DataStream Client', 'Vicon DataStream SDK' );

while ishandle( MessageBox )
    
    %Only look at unlabeled markers within 1m radius of origin
    if (abs(Xcoordinate) < 1000 && abs(Ycoordinate) < 1000)
        
        %Give test feedback when X coordinate of marker is > 20 from origin
        if (Xcoordinate > 100)
          Reaction1.motorON();
        else
          Reaction1.motorOFF();
        end
    
        %Starting experiment
        while (Start)

          if (Xcoordinate > Target)
              Reaction1.motorON();
          else
              Reaction1.motorOFF();
          end

          %Datalog Xcoordinate
          Xcoordinate = [Xcoordinate, new_X]

          if ((Target - new_X)< 5)
              TargetArea = [TargetArea, 1];
          else 
              TargetArea = [TargetArea, 0];
          end

          %If person is able to stay within target area for > 2seconds
          if (length(TargetArea)>200)
              %If last 200 have been 1's then finish
              if (min(TargetArea((length(TargetArea)+1-200):(length(TargetArea))))>0)
                  Start = false;
                  Reaction1.stopArduino();
              end
          end
          
        end
    end

end 


%Disconnect the Reaction Band
Reaction1.Disconnect();