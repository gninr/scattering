SetFactory("OpenCASCADE");

Point(1)  = {  2.0,  -2.0, 0, 1.0};
Point(2)  = {  2.0,   2.0, 0, 1.0};
Point(3)  = { -2.0,   2.0, 0, 1.0};
Point(4)  = { -2.0,  -2.0, 0, 1.0};
Point(5)  = { 2.25, -2.25, 0, 1.0};
Point(6)  = { 2.25,  2.25, 0, 1.0};
Point(7) = {-2.25,  2.25, 0, 1.0};
Point(8) = {-2.25, -2.25, 0, 1.0};

Circle(1) = {0, 0, 0, 1.0};
Line(2)  = {1, 2};
Line(3)  = {2, 3};
Line(4)  = {3, 4};
Line(5)  = {4, 1};
Line(6)  = {5, 6};
Line(7)  = {6, 7};
Line(8)  = {7, 8};
Line(9)  = {8, 5};

Curve Loop(1)  = {1};
Curve Loop(2)  = {2:5};
Curve Loop(3)  = {6:9};
Physical Curve("Gamma", 1) = {1};
Physical Curve("Gamma_I", 2) = {2:5};
Physical Curve("Gamma_D", 3) = {6:9};

Plane Surface(1) = {1, 2};
Plane Surface(2) = {2, 3};
Physical Surface("Omega_F", 1) = {1};
Physical Surface("Omega_A", 2) = {2};
