№а2
ж
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ви+

!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min

5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max

5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0

quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step

1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_conv2d/optimizer_step

/quant_conv2d/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_min

+quant_conv2d/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_max

+quant_conv2d/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_max*
_output_shapes
:*
dtype0

 quant_conv2d/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_min

4quant_conv2d/post_activation_min/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_min*
_output_shapes
: *
dtype0

 quant_conv2d/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_max

4quant_conv2d/post_activation_max/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_max*
_output_shapes
: *
dtype0

quant_conv2d_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_1/optimizer_step

1quant_conv2d_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_1/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_1/kernel_min

-quant_conv2d_1/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_1/kernel_max

-quant_conv2d_1/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_1/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_min

6quant_conv2d_1/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_1/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_1/post_activation_max

6quant_conv2d_1/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_1/post_activation_max*
_output_shapes
: *
dtype0

quant_conv2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_2/optimizer_step

1quant_conv2d_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_2/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_min

-quant_conv2d_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_max

-quant_conv2d_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_2/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_min

6quant_conv2d_2/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_2/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_max

6quant_conv2d_2/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_max*
_output_shapes
: *
dtype0

quant_conv2d_3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_3/optimizer_step

1quant_conv2d_3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_3/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_3/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_3/kernel_min

-quant_conv2d_3/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_3/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_3/kernel_max

-quant_conv2d_3/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_3/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_min

6quant_conv2d_3/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_3/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_3/post_activation_max

6quant_conv2d_3/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_3/post_activation_max*
_output_shapes
: *
dtype0

quant_conv2d_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_4/optimizer_step

1quant_conv2d_4/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_4/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_4/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_4/kernel_min

-quant_conv2d_4/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_4/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_4/kernel_max

-quant_conv2d_4/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_4/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_4/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_min

6quant_conv2d_4/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_4/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_4/post_activation_max

6quant_conv2d_4/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_4/post_activation_max*
_output_shapes
: *
dtype0

quant_conv2d_5/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_5/optimizer_step

1quant_conv2d_5/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_5/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_5/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_5/kernel_min

-quant_conv2d_5/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_5/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_5/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_5/kernel_max

-quant_conv2d_5/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_5/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_5/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_5/post_activation_min

6quant_conv2d_5/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_5/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_5/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_5/post_activation_max

6quant_conv2d_5/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_5/post_activation_max*
_output_shapes
: *
dtype0

quant_lambda/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_lambda/optimizer_step

/quant_lambda/optimizer_step/Read/ReadVariableOpReadVariableOpquant_lambda/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_6/optimizer_step

1quant_conv2d_6/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_6/optimizer_step*
_output_shapes
: *
dtype0

quant_conv2d_6/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_6/kernel_min

-quant_conv2d_6/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_6/kernel_min*
_output_shapes
:*
dtype0

quant_conv2d_6/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_6/kernel_max

-quant_conv2d_6/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_6/kernel_max*
_output_shapes
:*
dtype0

"quant_conv2d_6/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_6/post_activation_min

6quant_conv2d_6/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_6/post_activation_min*
_output_shapes
: *
dtype0

"quant_conv2d_6/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_6/post_activation_max

6quant_conv2d_6/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_6/post_activation_max*
_output_shapes
: *
dtype0

quant_add/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_add/optimizer_step
}
,quant_add/optimizer_step/Read/ReadVariableOpReadVariableOpquant_add/optimizer_step*
_output_shapes
: *
dtype0
|
quant_add/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namequant_add/output_min
u
(quant_add/output_min/Read/ReadVariableOpReadVariableOpquant_add/output_min*
_output_shapes
: *
dtype0
|
quant_add/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namequant_add/output_max
u
(quant_add/output_max/Read/ReadVariableOpReadVariableOpquant_add/output_max*
_output_shapes
: *
dtype0

quant_lambda_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_lambda_1/optimizer_step

1quant_lambda_1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_lambda_1/optimizer_step*
_output_shapes
: *
dtype0

quant_lambda_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_lambda_2/optimizer_step

1quant_lambda_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_lambda_2/optimizer_step*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ОЎ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ј­
valueэ­Bщ­ Bс­
з
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
Њ
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
regularization_losses
trainable_variables
	keras_api

	layer
optimizer_step
_weight_vars

kernel_min
 
kernel_max
!_quantize_activations
"post_activation_min
#post_activation_max
$_output_quantizers
%	variables
&regularization_losses
'trainable_variables
(	keras_api

	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers
2	variables
3regularization_losses
4trainable_variables
5	keras_api

	6layer
7optimizer_step
8_weight_vars
9
kernel_min
:
kernel_max
;_quantize_activations
<post_activation_min
=post_activation_max
>_output_quantizers
?	variables
@regularization_losses
Atrainable_variables
B	keras_api

	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api

	Player
Qoptimizer_step
R_weight_vars
S
kernel_min
T
kernel_max
U_quantize_activations
Vpost_activation_min
Wpost_activation_max
X_output_quantizers
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api

	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers
f	variables
gregularization_losses
htrainable_variables
i	keras_api
Ж
	jlayer
koptimizer_step
l_weight_vars
m_quantize_activations
n_output_quantizers
o	variables
pregularization_losses
qtrainable_variables
r	keras_api

	slayer
toptimizer_step
u_weight_vars
v
kernel_min
w
kernel_max
x_quantize_activations
ypost_activation_min
zpost_activation_max
{_output_quantizers
|	variables
}regularization_losses
~trainable_variables
	keras_api
ў

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars
	variables
regularization_losses
trainable_variables
	keras_api
П

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
П

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
љ
beta_1
beta_2

 decay
Ёlearning_rate
	Ђiter	Ѓmш	Єmщ	Ѕmъ	Іmы	Їmь	Јmэ	Љmю	Њmя	Ћm№	Ќmё	­mђ	Ўmѓ	Џmє	Аmѕ	Ѓvі	Єvї	Ѕvј	Іvљ	Їvњ	Јvћ	Љvќ	Њv§	Ћvў	Ќvџ	­v	Ўv	Џv	Аv
й
0
1
2
Ѓ3
Є4
5
6
 7
"8
#9
Ѕ10
І11
*12
,13
-14
/15
016
Ї17
Ј18
719
920
:21
<22
=23
Љ24
Њ25
D26
F27
G28
I29
J30
Ћ31
Ќ32
Q33
S34
T35
V36
W37
­38
Ў39
^40
`41
a42
c43
d44
k45
Џ46
А47
t48
v49
w50
y51
z52
53
54
55
56
57
 
t
Ѓ0
Є1
Ѕ2
І3
Ї4
Ј5
Љ6
Њ7
Ћ8
Ќ9
­10
Ў11
Џ12
А13
В
	variables
 Бlayer_regularization_losses
regularization_losses
Вnon_trainable_variables
trainable_variables
Гlayer_metrics
Дmetrics
Еlayers
 
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
qo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
В
	variables
 Жlayer_regularization_losses
regularization_losses
Зnon_trainable_variables
trainable_variables
Иlayer_metrics
Йmetrics
Кlayers
n
Ѓkernel
	Єbias
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
om
VARIABLE_VALUEquant_conv2d/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

П0
ge
VARIABLE_VALUEquant_conv2d/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEquant_conv2d/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
yw
VARIABLE_VALUE quant_conv2d/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE quant_conv2d/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Ѓ0
Є1
2
3
 4
"5
#6
 

Ѓ0
Є1
В
%	variables
 Рlayer_regularization_losses
&regularization_losses
Сnon_trainable_variables
'trainable_variables
Тlayer_metrics
Уmetrics
Фlayers
n
Ѕkernel
	Іbias
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
qo
VARIABLE_VALUEquant_conv2d_1/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

Щ0
ig
VARIABLE_VALUEquant_conv2d_1/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_1/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_1/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_1/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Ѕ0
І1
*2
,3
-4
/5
06
 

Ѕ0
І1
В
2	variables
 Ъlayer_regularization_losses
3regularization_losses
Ыnon_trainable_variables
4trainable_variables
Ьlayer_metrics
Эmetrics
Юlayers
n
Їkernel
	Јbias
Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
qo
VARIABLE_VALUEquant_conv2d_2/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

г0
ig
VARIABLE_VALUEquant_conv2d_2/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_2/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_2/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_2/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Ї0
Ј1
72
93
:4
<5
=6
 

Ї0
Ј1
В
?	variables
 дlayer_regularization_losses
@regularization_losses
еnon_trainable_variables
Atrainable_variables
жlayer_metrics
зmetrics
иlayers
n
Љkernel
	Њbias
й	variables
кregularization_losses
лtrainable_variables
м	keras_api
qo
VARIABLE_VALUEquant_conv2d_3/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

н0
ig
VARIABLE_VALUEquant_conv2d_3/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_3/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_3/post_activation_minClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_3/post_activation_maxClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Љ0
Њ1
D2
F3
G4
I5
J6
 

Љ0
Њ1
В
L	variables
 оlayer_regularization_losses
Mregularization_losses
пnon_trainable_variables
Ntrainable_variables
рlayer_metrics
сmetrics
тlayers
n
Ћkernel
	Ќbias
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
qo
VARIABLE_VALUEquant_conv2d_4/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

ч0
ig
VARIABLE_VALUEquant_conv2d_4/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_4/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_4/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_4/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Ћ0
Ќ1
Q2
S3
T4
V5
W6
 

Ћ0
Ќ1
В
Y	variables
 шlayer_regularization_losses
Zregularization_losses
щnon_trainable_variables
[trainable_variables
ъlayer_metrics
ыmetrics
ьlayers
n
­kernel
	Ўbias
э	variables
юregularization_losses
яtrainable_variables
№	keras_api
qo
VARIABLE_VALUEquant_conv2d_5/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

ё0
ig
VARIABLE_VALUEquant_conv2d_5/kernel_min:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_5/kernel_max:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_5/post_activation_minClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_5/post_activation_maxClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
­0
Ў1
^2
`3
a4
c5
d6
 

­0
Ў1
В
f	variables
 ђlayer_regularization_losses
gregularization_losses
ѓnon_trainable_variables
htrainable_variables
єlayer_metrics
ѕmetrics
іlayers
V
ї	variables
јregularization_losses
љtrainable_variables
њ	keras_api
om
VARIABLE_VALUEquant_lambda/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

k0
 
 
В
o	variables
 ћlayer_regularization_losses
pregularization_losses
ќnon_trainable_variables
qtrainable_variables
§layer_metrics
ўmetrics
џlayers
n
Џkernel
	Аbias
	variables
regularization_losses
trainable_variables
	keras_api
qo
VARIABLE_VALUEquant_conv2d_6/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
ig
VARIABLE_VALUEquant_conv2d_6/kernel_min:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_6/kernel_max:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_6/post_activation_minClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_6/post_activation_maxClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
Џ0
А1
t2
v3
w4
y5
z6
 

Џ0
А1
В
|	variables
 layer_regularization_losses
}regularization_losses
non_trainable_variables
~trainable_variables
layer_metrics
metrics
layers
V
	variables
regularization_losses
trainable_variables
	keras_api
lj
VARIABLE_VALUEquant_add/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
db
VARIABLE_VALUEquant_add/output_min:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEquant_add/output_max:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var

0
1
2
 
 
Е
	variables
 layer_regularization_losses
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
metrics
layers
V
	variables
regularization_losses
trainable_variables
	keras_api
rp
VARIABLE_VALUEquant_lambda_1/optimizer_step?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
 
 
Е
	variables
 layer_regularization_losses
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
metrics
layers
V
	variables
regularization_losses
trainable_variables
	keras_api
rp
VARIABLE_VALUEquant_lambda_2/optimizer_step?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
 
 
Е
	variables
  layer_regularization_losses
regularization_losses
Ёnon_trainable_variables
trainable_variables
Ђlayer_metrics
Ѓmetrics
Єlayers
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_4/kernel'variables/31/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_4/bias'variables/32/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_5/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_5/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_6/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_6/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE
 
л
0
1
2
3
4
 5
"6
#7
*8
,9
-10
/11
012
713
914
:15
<16
=17
D18
F19
G20
I21
J22
Q23
S24
T25
V26
W27
^28
`29
a30
c31
d32
k33
t34
v35
w36
y37
z38
39
40
41
42
43
 

Ѕ0
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 

0
1
2
 
 
 

Є0
 

Є0
Е
Л	variables
 Іlayer_regularization_losses
Мregularization_losses
Їnon_trainable_variables
Нtrainable_variables
Јlayer_metrics
Љmetrics
Њlayers

Ѓ0
Ћ2
 
#
0
1
 2
"3
#4
 
 

0

І0
 

І0
Е
Х	variables
 Ќlayer_regularization_losses
Цregularization_losses
­non_trainable_variables
Чtrainable_variables
Ўlayer_metrics
Џmetrics
Аlayers

Ѕ0
Б2
 
#
*0
,1
-2
/3
04
 
 

)0

Ј0
 

Ј0
Е
Я	variables
 Вlayer_regularization_losses
аregularization_losses
Гnon_trainable_variables
бtrainable_variables
Дlayer_metrics
Еmetrics
Жlayers

Ї0
З2
 
#
70
91
:2
<3
=4
 
 

60

Њ0
 

Њ0
Е
й	variables
 Иlayer_regularization_losses
кregularization_losses
Йnon_trainable_variables
лtrainable_variables
Кlayer_metrics
Лmetrics
Мlayers

Љ0
Н2
 
#
D0
F1
G2
I3
J4
 
 

C0

Ќ0
 

Ќ0
Е
у	variables
 Оlayer_regularization_losses
фregularization_losses
Пnon_trainable_variables
хtrainable_variables
Рlayer_metrics
Сmetrics
Тlayers

Ћ0
У2
 
#
Q0
S1
T2
V3
W4
 
 

P0

Ў0
 

Ў0
Е
э	variables
 Фlayer_regularization_losses
юregularization_losses
Хnon_trainable_variables
яtrainable_variables
Цlayer_metrics
Чmetrics
Шlayers

­0
Щ2
 
#
^0
`1
a2
c3
d4
 
 

]0
 
 
 
Е
ї	variables
 Ъlayer_regularization_losses
јregularization_losses
Ыnon_trainable_variables
љtrainable_variables
Ьlayer_metrics
Эmetrics
Юlayers
 

k0
 
 

j0

А0
 

А0
Е
	variables
 Яlayer_regularization_losses
regularization_losses
аnon_trainable_variables
trainable_variables
бlayer_metrics
вmetrics
гlayers

Џ0
д2
 
#
t0
v1
w2
y3
z4
 
 

s0
 
 
 
Е
	variables
 еlayer_regularization_losses
regularization_losses
жnon_trainable_variables
trainable_variables
зlayer_metrics
иmetrics
йlayers
 

0
1
2
 
 

0
 
 
 
Е
	variables
 кlayer_regularization_losses
regularization_losses
лnon_trainable_variables
trainable_variables
мlayer_metrics
нmetrics
оlayers
 

0
 
 

0
 
 
 
Е
	variables
 пlayer_regularization_losses
regularization_losses
рnon_trainable_variables
trainable_variables
сlayer_metrics
тmetrics
уlayers
 

0
 
 

0
8

фtotal

хcount
ц	variables
ч	keras_api
 
 
 
 
 

min_var
 max_var
 
 
 
 
 

,min_var
-max_var
 
 
 
 
 

9min_var
:max_var
 
 
 
 
 

Fmin_var
Gmax_var
 
 
 
 
 

Smin_var
Tmax_var
 
 
 
 
 

`min_var
amax_var
 
 
 
 
 
 
 
 
 
 

vmin_var
wmax_var
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ф0
х1

ц	variables
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_1/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_3/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_3/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_4/kernel/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_4/bias/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_5/kernel/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_5/bias/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_6/kernel/mCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_6/bias/mCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_1/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_3/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_3/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_4/kernel/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_4/bias/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_5/kernel/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_5/bias/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_6/kernel/vCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_6/bias/vCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ў
serving_default_input_1Placeholder*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*6
shape-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxconv2d/kernelquant_conv2d/kernel_minquant_conv2d/kernel_maxconv2d/bias quant_conv2d/post_activation_min quant_conv2d/post_activation_maxconv2d_1/kernelquant_conv2d_1/kernel_minquant_conv2d_1/kernel_maxconv2d_1/bias"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_maxconv2d_2/kernelquant_conv2d_2/kernel_minquant_conv2d_2/kernel_maxconv2d_2/bias"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxconv2d_3/kernelquant_conv2d_3/kernel_minquant_conv2d_3/kernel_maxconv2d_3/bias"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxconv2d_4/kernelquant_conv2d_4/kernel_minquant_conv2d_4/kernel_maxconv2d_4/bias"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_maxconv2d_5/kernelquant_conv2d_5/kernel_minquant_conv2d_5/kernel_maxconv2d_5/bias"quant_conv2d_5/post_activation_min"quant_conv2d_5/post_activation_maxconv2d_6/kernelquant_conv2d_6/kernel_minquant_conv2d_6/kernel_maxconv2d_6/bias"quant_conv2d_6/post_activation_min"quant_conv2d_6/post_activation_maxquant_add/output_minquant_add/output_max*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_2122790
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 #
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5quantize_layer/quantize_layer_min/Read/ReadVariableOp5quantize_layer/quantize_layer_max/Read/ReadVariableOp1quantize_layer/optimizer_step/Read/ReadVariableOp/quant_conv2d/optimizer_step/Read/ReadVariableOp+quant_conv2d/kernel_min/Read/ReadVariableOp+quant_conv2d/kernel_max/Read/ReadVariableOp4quant_conv2d/post_activation_min/Read/ReadVariableOp4quant_conv2d/post_activation_max/Read/ReadVariableOp1quant_conv2d_1/optimizer_step/Read/ReadVariableOp-quant_conv2d_1/kernel_min/Read/ReadVariableOp-quant_conv2d_1/kernel_max/Read/ReadVariableOp6quant_conv2d_1/post_activation_min/Read/ReadVariableOp6quant_conv2d_1/post_activation_max/Read/ReadVariableOp1quant_conv2d_2/optimizer_step/Read/ReadVariableOp-quant_conv2d_2/kernel_min/Read/ReadVariableOp-quant_conv2d_2/kernel_max/Read/ReadVariableOp6quant_conv2d_2/post_activation_min/Read/ReadVariableOp6quant_conv2d_2/post_activation_max/Read/ReadVariableOp1quant_conv2d_3/optimizer_step/Read/ReadVariableOp-quant_conv2d_3/kernel_min/Read/ReadVariableOp-quant_conv2d_3/kernel_max/Read/ReadVariableOp6quant_conv2d_3/post_activation_min/Read/ReadVariableOp6quant_conv2d_3/post_activation_max/Read/ReadVariableOp1quant_conv2d_4/optimizer_step/Read/ReadVariableOp-quant_conv2d_4/kernel_min/Read/ReadVariableOp-quant_conv2d_4/kernel_max/Read/ReadVariableOp6quant_conv2d_4/post_activation_min/Read/ReadVariableOp6quant_conv2d_4/post_activation_max/Read/ReadVariableOp1quant_conv2d_5/optimizer_step/Read/ReadVariableOp-quant_conv2d_5/kernel_min/Read/ReadVariableOp-quant_conv2d_5/kernel_max/Read/ReadVariableOp6quant_conv2d_5/post_activation_min/Read/ReadVariableOp6quant_conv2d_5/post_activation_max/Read/ReadVariableOp/quant_lambda/optimizer_step/Read/ReadVariableOp1quant_conv2d_6/optimizer_step/Read/ReadVariableOp-quant_conv2d_6/kernel_min/Read/ReadVariableOp-quant_conv2d_6/kernel_max/Read/ReadVariableOp6quant_conv2d_6/post_activation_min/Read/ReadVariableOp6quant_conv2d_6/post_activation_max/Read/ReadVariableOp,quant_add/optimizer_step/Read/ReadVariableOp(quant_add/output_min/Read/ReadVariableOp(quant_add/output_max/Read/ReadVariableOp1quant_lambda_1/optimizer_step/Read/ReadVariableOp1quant_lambda_2/optimizer_step/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOpConst*j
Tinc
a2_	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_2124722
з
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv2d/optimizer_stepquant_conv2d/kernel_minquant_conv2d/kernel_max quant_conv2d/post_activation_min quant_conv2d/post_activation_maxquant_conv2d_1/optimizer_stepquant_conv2d_1/kernel_minquant_conv2d_1/kernel_max"quant_conv2d_1/post_activation_min"quant_conv2d_1/post_activation_maxquant_conv2d_2/optimizer_stepquant_conv2d_2/kernel_minquant_conv2d_2/kernel_max"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxquant_conv2d_3/optimizer_stepquant_conv2d_3/kernel_minquant_conv2d_3/kernel_max"quant_conv2d_3/post_activation_min"quant_conv2d_3/post_activation_maxquant_conv2d_4/optimizer_stepquant_conv2d_4/kernel_minquant_conv2d_4/kernel_max"quant_conv2d_4/post_activation_min"quant_conv2d_4/post_activation_maxquant_conv2d_5/optimizer_stepquant_conv2d_5/kernel_minquant_conv2d_5/kernel_max"quant_conv2d_5/post_activation_min"quant_conv2d_5/post_activation_maxquant_lambda/optimizer_stepquant_conv2d_6/optimizer_stepquant_conv2d_6/kernel_minquant_conv2d_6/kernel_max"quant_conv2d_6/post_activation_min"quant_conv2d_6/post_activation_maxquant_add/optimizer_stepquant_add/output_minquant_add/output_maxquant_lambda_1/optimizer_stepquant_lambda_2/optimizer_stepbeta_1beta_2decaylearning_rate	Adam/iterconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biastotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v*i
Tinb
`2^*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_2125011ЗЁ(
њ
g
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2121211

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
DepthToSpace
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В	

0__inference_quant_conv2d_6_layer_call_fn_2124316

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21214702
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2121763

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124126

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь'
ў
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123518

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identityЂ#AllValuesQuantize/AssignMaxAllValueЂ#AllValuesQuantize/AssignMinAllValueЂ8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ(AllValuesQuantize/Maximum/ReadVariableOpЂ(AllValuesQuantize/Minimum/ReadVariableOp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMaxО
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOpЙ
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y­
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1О
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOpЙ
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y­
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Т
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)AllValuesQuantize/FakeQuantWithMinMaxVarsЛ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е


'__inference_model_layer_call_fn_2123391

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_21212242
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В	

0__inference_quant_conv2d_3_layer_call_fn_2123952

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21217632
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
пS

B__inference_model_layer_call_and_return_conditional_losses_2122685
input_1 
quantize_layer_2122580:  
quantize_layer_2122582: .
quant_conv2d_2122585:"
quant_conv2d_2122587:"
quant_conv2d_2122589:"
quant_conv2d_2122591:
quant_conv2d_2122593: 
quant_conv2d_2122595: 0
quant_conv2d_1_2122598:$
quant_conv2d_1_2122600:$
quant_conv2d_1_2122602:$
quant_conv2d_1_2122604: 
quant_conv2d_1_2122606:  
quant_conv2d_1_2122608: 0
quant_conv2d_2_2122611:$
quant_conv2d_2_2122613:$
quant_conv2d_2_2122615:$
quant_conv2d_2_2122617: 
quant_conv2d_2_2122619:  
quant_conv2d_2_2122621: 0
quant_conv2d_3_2122624:$
quant_conv2d_3_2122626:$
quant_conv2d_3_2122628:$
quant_conv2d_3_2122630: 
quant_conv2d_3_2122632:  
quant_conv2d_3_2122634: 0
quant_conv2d_4_2122637:$
quant_conv2d_4_2122639:$
quant_conv2d_4_2122641:$
quant_conv2d_4_2122643: 
quant_conv2d_4_2122645:  
quant_conv2d_4_2122647: 0
quant_conv2d_5_2122650:$
quant_conv2d_5_2122652:$
quant_conv2d_5_2122654:$
quant_conv2d_5_2122656: 
quant_conv2d_5_2122658:  
quant_conv2d_5_2122660: 0
quant_conv2d_6_2122664:$
quant_conv2d_6_2122666:$
quant_conv2d_6_2122668:$
quant_conv2d_6_2122670: 
quant_conv2d_6_2122672:  
quant_conv2d_6_2122674: 
quant_add_2122677: 
quant_add_2122679: 
identityЂ!quant_add/StatefulPartitionedCallЂ$quant_conv2d/StatefulPartitionedCallЂ&quant_conv2d_1/StatefulPartitionedCallЂ&quant_conv2d_2/StatefulPartitionedCallЂ&quant_conv2d_3/StatefulPartitionedCallЂ&quant_conv2d_4/StatefulPartitionedCallЂ&quant_conv2d_5/StatefulPartitionedCallЂ&quant_conv2d_6/StatefulPartitionedCallЂ&quantize_layer/StatefulPartitionedCallб
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_2122580quantize_layer_2122582*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21220632(
&quantize_layer/StatefulPartitionedCallЯ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_2122585quant_conv2d_2122587quant_conv2d_2122589quant_conv2d_2122591quant_conv2d_2122593quant_conv2d_2122595*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21220152&
$quant_conv2d/StatefulPartitionedCallп
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_2122598quant_conv2d_1_2122600quant_conv2d_1_2122602quant_conv2d_1_2122604quant_conv2d_1_2122606quant_conv2d_1_2122608*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21219312(
&quant_conv2d_1/StatefulPartitionedCallс
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_2122611quant_conv2d_2_2122613quant_conv2d_2_2122615quant_conv2d_2_2122617quant_conv2d_2_2122619quant_conv2d_2_2122621*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21218472(
&quant_conv2d_2/StatefulPartitionedCallс
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_2122624quant_conv2d_3_2122626quant_conv2d_3_2122628quant_conv2d_3_2122630quant_conv2d_3_2122632quant_conv2d_3_2122634*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21217632(
&quant_conv2d_3/StatefulPartitionedCallс
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_2122637quant_conv2d_4_2122639quant_conv2d_4_2122641quant_conv2d_4_2122643quant_conv2d_4_2122645quant_conv2d_4_2122647*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21216792(
&quant_conv2d_4/StatefulPartitionedCallс
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_2122650quant_conv2d_5_2122652quant_conv2d_5_2122654quant_conv2d_5_2122656quant_conv2d_5_2122658quant_conv2d_5_2122660*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21215952(
&quant_conv2d_5/StatefulPartitionedCallЙ
quant_lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21215232
quant_lambda/PartitionedCallс
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_2122664quant_conv2d_6_2122666quant_conv2d_6_2122668quant_conv2d_6_2122670quant_conv2d_6_2122672quant_conv2d_6_2122674*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21214702(
&quant_conv2d_6/StatefulPartitionedCall
!quant_add/StatefulPartitionedCallStatefulPartitionedCall%quant_lambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_2122677quant_add_2122679*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21213952#
!quant_add/StatefulPartitionedCallЊ
quant_lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21213512 
quant_lambda_1/PartitionedCallЇ
quant_lambda_2/PartitionedCallPartitionedCall'quant_lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21213352 
quant_lambda_2/PartitionedCallџ
IdentityIdentity'quant_lambda_2/PartitionedCall:output:0"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1
В	

0__inference_quant_conv2d_2_layer_call_fn_2123848

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21218472
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И


'__inference_model_layer_call_fn_2121319
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_21212242
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1
Ў	
џ
.__inference_quant_conv2d_layer_call_fn_2123640

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21220152
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
T

B__inference_model_layer_call_and_return_conditional_losses_2121224

inputs 
quantize_layer_2120924:  
quantize_layer_2120926: .
quant_conv2d_2120951:"
quant_conv2d_2120953:"
quant_conv2d_2120955:"
quant_conv2d_2120957:
quant_conv2d_2120959: 
quant_conv2d_2120961: 0
quant_conv2d_1_2120986:$
quant_conv2d_1_2120988:$
quant_conv2d_1_2120990:$
quant_conv2d_1_2120992: 
quant_conv2d_1_2120994:  
quant_conv2d_1_2120996: 0
quant_conv2d_2_2121021:$
quant_conv2d_2_2121023:$
quant_conv2d_2_2121025:$
quant_conv2d_2_2121027: 
quant_conv2d_2_2121029:  
quant_conv2d_2_2121031: 0
quant_conv2d_3_2121056:$
quant_conv2d_3_2121058:$
quant_conv2d_3_2121060:$
quant_conv2d_3_2121062: 
quant_conv2d_3_2121064:  
quant_conv2d_3_2121066: 0
quant_conv2d_4_2121091:$
quant_conv2d_4_2121093:$
quant_conv2d_4_2121095:$
quant_conv2d_4_2121097: 
quant_conv2d_4_2121099:  
quant_conv2d_4_2121101: 0
quant_conv2d_5_2121126:$
quant_conv2d_5_2121128:$
quant_conv2d_5_2121130:$
quant_conv2d_5_2121132: 
quant_conv2d_5_2121134:  
quant_conv2d_5_2121136: 0
quant_conv2d_6_2121176:$
quant_conv2d_6_2121178:$
quant_conv2d_6_2121180:$
quant_conv2d_6_2121182: 
quant_conv2d_6_2121184:  
quant_conv2d_6_2121186: 
quant_add_2121201: 
quant_add_2121203: 
identityЂ!quant_add/StatefulPartitionedCallЂ$quant_conv2d/StatefulPartitionedCallЂ&quant_conv2d_1/StatefulPartitionedCallЂ&quant_conv2d_2/StatefulPartitionedCallЂ&quant_conv2d_3/StatefulPartitionedCallЂ&quant_conv2d_4/StatefulPartitionedCallЂ&quant_conv2d_5/StatefulPartitionedCallЂ&quant_conv2d_6/StatefulPartitionedCallЂ&quantize_layer/StatefulPartitionedCallд
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_2120924quantize_layer_2120926*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21209232(
&quantize_layer/StatefulPartitionedCallг
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_2120951quant_conv2d_2120953quant_conv2d_2120955quant_conv2d_2120957quant_conv2d_2120959quant_conv2d_2120961*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21209502&
$quant_conv2d/StatefulPartitionedCallу
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_2120986quant_conv2d_1_2120988quant_conv2d_1_2120990quant_conv2d_1_2120992quant_conv2d_1_2120994quant_conv2d_1_2120996*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21209852(
&quant_conv2d_1/StatefulPartitionedCallх
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_2121021quant_conv2d_2_2121023quant_conv2d_2_2121025quant_conv2d_2_2121027quant_conv2d_2_2121029quant_conv2d_2_2121031*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21210202(
&quant_conv2d_2/StatefulPartitionedCallх
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_2121056quant_conv2d_3_2121058quant_conv2d_3_2121060quant_conv2d_3_2121062quant_conv2d_3_2121064quant_conv2d_3_2121066*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21210552(
&quant_conv2d_3/StatefulPartitionedCallх
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_2121091quant_conv2d_4_2121093quant_conv2d_4_2121095quant_conv2d_4_2121097quant_conv2d_4_2121099quant_conv2d_4_2121101*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21210902(
&quant_conv2d_4/StatefulPartitionedCallх
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_2121126quant_conv2d_5_2121128quant_conv2d_5_2121130quant_conv2d_5_2121132quant_conv2d_5_2121134quant_conv2d_5_2121136*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21211252(
&quant_conv2d_5/StatefulPartitionedCallЙ
quant_lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21211532
quant_lambda/PartitionedCallх
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_2121176quant_conv2d_6_2121178quant_conv2d_6_2121180quant_conv2d_6_2121182quant_conv2d_6_2121184quant_conv2d_6_2121186*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21211752(
&quant_conv2d_6/StatefulPartitionedCall
!quant_add/StatefulPartitionedCallStatefulPartitionedCall%quant_lambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_2121201quant_add_2121203*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21212002#
!quant_add/StatefulPartitionedCallЊ
quant_lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21212112 
quant_lambda_1/PartitionedCallЇ
quant_lambda_2/PartitionedCallPartitionedCall'quant_lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21212212 
quant_lambda_2/PartitionedCallџ
IdentityIdentity'quant_lambda_2/PartitionedCall:output:0"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ют
ѓ<
B__inference_model_layer_call_and_return_conditional_losses_2122930

inputsZ
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: o
Uquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource::
,quant_conv2d_biasadd_readvariableop_resource:X
Nquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Z
Pquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_1_biasadd_readvariableop_resource:Z
Pquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:Z
Pquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_3_biasadd_readvariableop_resource:Z
Pquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_4_biasadd_readvariableop_resource:Z
Pquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_5_biasadd_readvariableop_resource:Z
Pquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_6_biasadd_readvariableop_resource:Z
Pquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: U
Kquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: W
Mquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂDquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ#quant_conv2d/BiasAdd/ReadVariableOpЂLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂNquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂNquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂGquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_1/BiasAdd/ReadVariableOpЂNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_2/BiasAdd/ReadVariableOpЂNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_3/BiasAdd/ReadVariableOpЂNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_4/BiasAdd/ReadVariableOpЂNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_5/BiasAdd/ReadVariableOpЂNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_6/BiasAdd/ReadVariableOpЂNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ЂGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ў
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsК
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpUquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02N
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpД
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Д
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2х
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЕ
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d/Conv2DГ
#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#quant_conv2d/BiasAdd/ReadVariableOpЮ
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d/BiasAdd
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d/Relu
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpNquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02G
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpPquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ28
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsР
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЙ
quant_conv2d_1/Conv2DConv2D@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_1/Conv2DЙ
%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_1/BiasAdd/ReadVariableOpж
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_1/BiasAdd
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_1/Relu
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsР
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_2/Conv2DConv2DBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_2/Conv2DЙ
%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_2/BiasAdd/ReadVariableOpж
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_2/BiasAdd
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_2/Relu
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsР
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_3/Conv2DЙ
%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_3/BiasAdd/ReadVariableOpж
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_3/BiasAdd
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_3/Relu
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsР
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_4/Conv2DЙ
%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_4/BiasAdd/ReadVariableOpж
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_4/BiasAdd
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_4/Relu
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsР
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_5/Conv2DConv2DBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_5/Conv2DЙ
%quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_5/BiasAdd/ReadVariableOpж
quant_conv2d_5/BiasAddBiasAddquant_conv2d_5/Conv2D:output:0-quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_5/BiasAdd
quant_conv2d_5/ReluReluquant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_5/Relu
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_5/Relu:activations:0Oquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsv
quant_lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
quant_lambda/concat/axis
quant_lambda/concatConcatV2Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0!quant_lambda/concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_lambda/concatР
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpК
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1К
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_6/Conv2DConv2DBquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_6/Conv2DЙ
%quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_6/BiasAdd/ReadVariableOpж
quant_conv2d_6/BiasAddBiasAddquant_conv2d_6/Conv2D:output:0-quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_6/BiasAdd
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЁ
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d_6/BiasAdd:output:0Oquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsе
quant_add/addAddV2quant_lambda/concat:output:0Bquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_add/add
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpKquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02D
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpMquant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ѕ
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_add/add:z:0Jquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsч
quant_lambda_1/DepthToSpaceDepthToSpace=quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
quant_lambda_1/DepthToSpace
&quant_lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2(
&quant_lambda_2/clip_by_value/Minimum/yњ
$quant_lambda_2/clip_by_value/MinimumMinimum$quant_lambda_1/DepthToSpace:output:0/quant_lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2&
$quant_lambda_2/clip_by_value/Minimum
quant_lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
quant_lambda_2/clip_by_value/yц
quant_lambda_2/clip_by_valueMaximum(quant_lambda_2/clip_by_value/Minimum:z:0'quant_lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_lambda_2/clip_by_value
IdentityIdentity quant_lambda_2/clip_by_value:z:0C^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1$^quant_conv2d/BiasAdd/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2F^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_5/BiasAdd/ReadVariableOpO^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_6/BiasAdd/ReadVariableOpO^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1H^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_5/BiasAdd/ReadVariableOp%quant_conv2d_5/BiasAdd/ReadVariableOp2 
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_6/BiasAdd/ReadVariableOp%quant_conv2d_6/BiasAdd/ReadVariableOp2 
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


0__inference_quantize_layer_layer_call_fn_2123536

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21220632
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж
М
.__inference_quant_lambda_layer_call_fn_2124201
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identityР
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21211532
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/3:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/4:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/5:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/6:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/7:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/8
И
L
0__inference_quant_lambda_1_layer_call_fn_2124389

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21212112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј]
	
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124282

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ь
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124077

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ2
Л
F__inference_quant_add_layer_call_and_return_conditional_losses_2124354
inputs_0
inputs_1@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1s
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
add
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinadd:z:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxadd:z:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1У
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsу
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
№'
Ј
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123869

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж	

0__inference_quant_conv2d_4_layer_call_fn_2124039

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21210902
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ&
Ј
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2121175

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ь
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
g
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2121221

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
clip_by_value/Minimum/yЏ
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yЊ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
g
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124402

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
clip_by_value/Minimum/yЏ
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yЊ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
е
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2121153

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisп
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
мS

B__inference_model_layer_call_and_return_conditional_losses_2122277

inputs 
quantize_layer_2122172:  
quantize_layer_2122174: .
quant_conv2d_2122177:"
quant_conv2d_2122179:"
quant_conv2d_2122181:"
quant_conv2d_2122183:
quant_conv2d_2122185: 
quant_conv2d_2122187: 0
quant_conv2d_1_2122190:$
quant_conv2d_1_2122192:$
quant_conv2d_1_2122194:$
quant_conv2d_1_2122196: 
quant_conv2d_1_2122198:  
quant_conv2d_1_2122200: 0
quant_conv2d_2_2122203:$
quant_conv2d_2_2122205:$
quant_conv2d_2_2122207:$
quant_conv2d_2_2122209: 
quant_conv2d_2_2122211:  
quant_conv2d_2_2122213: 0
quant_conv2d_3_2122216:$
quant_conv2d_3_2122218:$
quant_conv2d_3_2122220:$
quant_conv2d_3_2122222: 
quant_conv2d_3_2122224:  
quant_conv2d_3_2122226: 0
quant_conv2d_4_2122229:$
quant_conv2d_4_2122231:$
quant_conv2d_4_2122233:$
quant_conv2d_4_2122235: 
quant_conv2d_4_2122237:  
quant_conv2d_4_2122239: 0
quant_conv2d_5_2122242:$
quant_conv2d_5_2122244:$
quant_conv2d_5_2122246:$
quant_conv2d_5_2122248: 
quant_conv2d_5_2122250:  
quant_conv2d_5_2122252: 0
quant_conv2d_6_2122256:$
quant_conv2d_6_2122258:$
quant_conv2d_6_2122260:$
quant_conv2d_6_2122262: 
quant_conv2d_6_2122264:  
quant_conv2d_6_2122266: 
quant_add_2122269: 
quant_add_2122271: 
identityЂ!quant_add/StatefulPartitionedCallЂ$quant_conv2d/StatefulPartitionedCallЂ&quant_conv2d_1/StatefulPartitionedCallЂ&quant_conv2d_2/StatefulPartitionedCallЂ&quant_conv2d_3/StatefulPartitionedCallЂ&quant_conv2d_4/StatefulPartitionedCallЂ&quant_conv2d_5/StatefulPartitionedCallЂ&quant_conv2d_6/StatefulPartitionedCallЂ&quantize_layer/StatefulPartitionedCallа
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_2122172quantize_layer_2122174*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21220632(
&quantize_layer/StatefulPartitionedCallЯ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_2122177quant_conv2d_2122179quant_conv2d_2122181quant_conv2d_2122183quant_conv2d_2122185quant_conv2d_2122187*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21220152&
$quant_conv2d/StatefulPartitionedCallп
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_2122190quant_conv2d_1_2122192quant_conv2d_1_2122194quant_conv2d_1_2122196quant_conv2d_1_2122198quant_conv2d_1_2122200*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21219312(
&quant_conv2d_1/StatefulPartitionedCallс
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_2122203quant_conv2d_2_2122205quant_conv2d_2_2122207quant_conv2d_2_2122209quant_conv2d_2_2122211quant_conv2d_2_2122213*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21218472(
&quant_conv2d_2/StatefulPartitionedCallс
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_2122216quant_conv2d_3_2122218quant_conv2d_3_2122220quant_conv2d_3_2122222quant_conv2d_3_2122224quant_conv2d_3_2122226*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21217632(
&quant_conv2d_3/StatefulPartitionedCallс
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_2122229quant_conv2d_4_2122231quant_conv2d_4_2122233quant_conv2d_4_2122235quant_conv2d_4_2122237quant_conv2d_4_2122239*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21216792(
&quant_conv2d_4/StatefulPartitionedCallс
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_2122242quant_conv2d_5_2122244quant_conv2d_5_2122246quant_conv2d_5_2122248quant_conv2d_5_2122250quant_conv2d_5_2122252*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21215952(
&quant_conv2d_5/StatefulPartitionedCallЙ
quant_lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21215232
quant_lambda/PartitionedCallс
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_2122256quant_conv2d_6_2122258quant_conv2d_6_2122260quant_conv2d_6_2122262quant_conv2d_6_2122264quant_conv2d_6_2122266*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21214702(
&quant_conv2d_6/StatefulPartitionedCall
!quant_add/StatefulPartitionedCallStatefulPartitionedCall%quant_lambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_2122269quant_add_2122271*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21213952#
!quant_add/StatefulPartitionedCallЊ
quant_lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21213512 
quant_lambda_1/PartitionedCallЇ
quant_lambda_2/PartitionedCallPartitionedCall'quant_lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21213352 
quant_lambda_2/PartitionedCallџ
IdentityIdentity'quant_lambda_2/PartitionedCall:output:0"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж
М
.__inference_quant_lambda_layer_call_fn_2124214
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identityР
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21215232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/3:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/4:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/5:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/6:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/7:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/8
Ж	

0__inference_quant_conv2d_3_layer_call_fn_2123935

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21210552
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs



'__inference_model_layer_call_fn_2123488

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	!$'**2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_21222772
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2124022

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Й	

+__inference_quant_add_layer_call_fn_2124364
inputs_0
inputs_1
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21212002
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
В	

0__inference_quant_conv2d_4_layer_call_fn_2124056

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21216792
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В	

0__inference_quant_conv2d_5_layer_call_fn_2124160

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21215952
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


0__inference_quantize_layer_layer_call_fn_2123527

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21209232
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж	

0__inference_quant_conv2d_6_layer_call_fn_2124299

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21211752
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ю'
І
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123557

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь'
ў
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2122063

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identityЂ#AllValuesQuantize/AssignMaxAllValueЂ#AllValuesQuantize/AssignMinAllValueЂ8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ(AllValuesQuantize/Maximum/ReadVariableOpЂ(AllValuesQuantize/Minimum/ReadVariableOp
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMaxО
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOpЙ
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y­
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1О
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOpЙ
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y­
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Т
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)AllValuesQuantize/FakeQuantWithMinMaxVarsЛ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В	
џ
.__inference_quant_conv2d_layer_call_fn_2123623

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21209502
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs



'__inference_model_layer_call_fn_2122469
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	!$'**2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_21222772
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1



%__inference_signature_wrapper_2122790
input_1
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: #
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: $

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: $

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: $

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29: 

unknown_30: $

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35: 

unknown_36: $

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41: 

unknown_42: 

unknown_43: 

unknown_44: 
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_21209072
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1
Е	

+__inference_quant_add_layer_call_fn_2124374
inputs_0
inputs_1
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21213952
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
ђ^
	
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123710

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2121931

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д
ў
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123497

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂ8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ю
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Т
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)AllValuesQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:09^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123814

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь
а;
#__inference__traced_restore_2125011
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 8
.assignvariableop_3_quant_conv2d_optimizer_step: 8
*assignvariableop_4_quant_conv2d_kernel_min:8
*assignvariableop_5_quant_conv2d_kernel_max:=
3assignvariableop_6_quant_conv2d_post_activation_min: =
3assignvariableop_7_quant_conv2d_post_activation_max: :
0assignvariableop_8_quant_conv2d_1_optimizer_step: :
,assignvariableop_9_quant_conv2d_1_kernel_min:;
-assignvariableop_10_quant_conv2d_1_kernel_max:@
6assignvariableop_11_quant_conv2d_1_post_activation_min: @
6assignvariableop_12_quant_conv2d_1_post_activation_max: ;
1assignvariableop_13_quant_conv2d_2_optimizer_step: ;
-assignvariableop_14_quant_conv2d_2_kernel_min:;
-assignvariableop_15_quant_conv2d_2_kernel_max:@
6assignvariableop_16_quant_conv2d_2_post_activation_min: @
6assignvariableop_17_quant_conv2d_2_post_activation_max: ;
1assignvariableop_18_quant_conv2d_3_optimizer_step: ;
-assignvariableop_19_quant_conv2d_3_kernel_min:;
-assignvariableop_20_quant_conv2d_3_kernel_max:@
6assignvariableop_21_quant_conv2d_3_post_activation_min: @
6assignvariableop_22_quant_conv2d_3_post_activation_max: ;
1assignvariableop_23_quant_conv2d_4_optimizer_step: ;
-assignvariableop_24_quant_conv2d_4_kernel_min:;
-assignvariableop_25_quant_conv2d_4_kernel_max:@
6assignvariableop_26_quant_conv2d_4_post_activation_min: @
6assignvariableop_27_quant_conv2d_4_post_activation_max: ;
1assignvariableop_28_quant_conv2d_5_optimizer_step: ;
-assignvariableop_29_quant_conv2d_5_kernel_min:;
-assignvariableop_30_quant_conv2d_5_kernel_max:@
6assignvariableop_31_quant_conv2d_5_post_activation_min: @
6assignvariableop_32_quant_conv2d_5_post_activation_max: 9
/assignvariableop_33_quant_lambda_optimizer_step: ;
1assignvariableop_34_quant_conv2d_6_optimizer_step: ;
-assignvariableop_35_quant_conv2d_6_kernel_min:;
-assignvariableop_36_quant_conv2d_6_kernel_max:@
6assignvariableop_37_quant_conv2d_6_post_activation_min: @
6assignvariableop_38_quant_conv2d_6_post_activation_max: 6
,assignvariableop_39_quant_add_optimizer_step: 2
(assignvariableop_40_quant_add_output_min: 2
(assignvariableop_41_quant_add_output_max: ;
1assignvariableop_42_quant_lambda_1_optimizer_step: ;
1assignvariableop_43_quant_lambda_2_optimizer_step: $
assignvariableop_44_beta_1: $
assignvariableop_45_beta_2: #
assignvariableop_46_decay: +
!assignvariableop_47_learning_rate: '
assignvariableop_48_adam_iter:	 ;
!assignvariableop_49_conv2d_kernel:-
assignvariableop_50_conv2d_bias:=
#assignvariableop_51_conv2d_1_kernel:/
!assignvariableop_52_conv2d_1_bias:=
#assignvariableop_53_conv2d_2_kernel:/
!assignvariableop_54_conv2d_2_bias:=
#assignvariableop_55_conv2d_3_kernel:/
!assignvariableop_56_conv2d_3_bias:=
#assignvariableop_57_conv2d_4_kernel:/
!assignvariableop_58_conv2d_4_bias:=
#assignvariableop_59_conv2d_5_kernel:/
!assignvariableop_60_conv2d_5_bias:=
#assignvariableop_61_conv2d_6_kernel:/
!assignvariableop_62_conv2d_6_bias:#
assignvariableop_63_total: #
assignvariableop_64_count: B
(assignvariableop_65_adam_conv2d_kernel_m:4
&assignvariableop_66_adam_conv2d_bias_m:D
*assignvariableop_67_adam_conv2d_1_kernel_m:6
(assignvariableop_68_adam_conv2d_1_bias_m:D
*assignvariableop_69_adam_conv2d_2_kernel_m:6
(assignvariableop_70_adam_conv2d_2_bias_m:D
*assignvariableop_71_adam_conv2d_3_kernel_m:6
(assignvariableop_72_adam_conv2d_3_bias_m:D
*assignvariableop_73_adam_conv2d_4_kernel_m:6
(assignvariableop_74_adam_conv2d_4_bias_m:D
*assignvariableop_75_adam_conv2d_5_kernel_m:6
(assignvariableop_76_adam_conv2d_5_bias_m:D
*assignvariableop_77_adam_conv2d_6_kernel_m:6
(assignvariableop_78_adam_conv2d_6_bias_m:B
(assignvariableop_79_adam_conv2d_kernel_v:4
&assignvariableop_80_adam_conv2d_bias_v:D
*assignvariableop_81_adam_conv2d_1_kernel_v:6
(assignvariableop_82_adam_conv2d_1_bias_v:D
*assignvariableop_83_adam_conv2d_2_kernel_v:6
(assignvariableop_84_adam_conv2d_2_bias_v:D
*assignvariableop_85_adam_conv2d_3_kernel_v:6
(assignvariableop_86_adam_conv2d_3_bias_v:D
*assignvariableop_87_adam_conv2d_4_kernel_v:6
(assignvariableop_88_adam_conv2d_4_bias_v:D
*assignvariableop_89_adam_conv2d_5_kernel_v:6
(assignvariableop_90_adam_conv2d_5_bias_v:D
*assignvariableop_91_adam_conv2d_6_kernel_v:6
(assignvariableop_92_adam_conv2d_6_bias_v:
identity_94ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92Ь-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*и,
valueЮ,BЫ,^BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЭ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*б
valueЧBФ^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesћ
ј::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*l
dtypesb
`2^	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityБ
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Й
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Е
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Г
AssignVariableOp_3AssignVariableOp.assignvariableop_3_quant_conv2d_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Џ
AssignVariableOp_4AssignVariableOp*assignvariableop_4_quant_conv2d_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Џ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_quant_conv2d_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6И
AssignVariableOp_6AssignVariableOp3assignvariableop_6_quant_conv2d_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7И
AssignVariableOp_7AssignVariableOp3assignvariableop_7_quant_conv2d_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Е
AssignVariableOp_8AssignVariableOp0assignvariableop_8_quant_conv2d_1_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOp,assignvariableop_9_quant_conv2d_1_kernel_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Е
AssignVariableOp_10AssignVariableOp-assignvariableop_10_quant_conv2d_1_kernel_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11О
AssignVariableOp_11AssignVariableOp6assignvariableop_11_quant_conv2d_1_post_activation_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12О
AssignVariableOp_12AssignVariableOp6assignvariableop_12_quant_conv2d_1_post_activation_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Й
AssignVariableOp_13AssignVariableOp1assignvariableop_13_quant_conv2d_2_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Е
AssignVariableOp_14AssignVariableOp-assignvariableop_14_quant_conv2d_2_kernel_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Е
AssignVariableOp_15AssignVariableOp-assignvariableop_15_quant_conv2d_2_kernel_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16О
AssignVariableOp_16AssignVariableOp6assignvariableop_16_quant_conv2d_2_post_activation_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17О
AssignVariableOp_17AssignVariableOp6assignvariableop_17_quant_conv2d_2_post_activation_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Й
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_conv2d_3_optimizer_stepIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Е
AssignVariableOp_19AssignVariableOp-assignvariableop_19_quant_conv2d_3_kernel_minIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp-assignvariableop_20_quant_conv2d_3_kernel_maxIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21О
AssignVariableOp_21AssignVariableOp6assignvariableop_21_quant_conv2d_3_post_activation_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22О
AssignVariableOp_22AssignVariableOp6assignvariableop_22_quant_conv2d_3_post_activation_maxIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Й
AssignVariableOp_23AssignVariableOp1assignvariableop_23_quant_conv2d_4_optimizer_stepIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Е
AssignVariableOp_24AssignVariableOp-assignvariableop_24_quant_conv2d_4_kernel_minIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Е
AssignVariableOp_25AssignVariableOp-assignvariableop_25_quant_conv2d_4_kernel_maxIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26О
AssignVariableOp_26AssignVariableOp6assignvariableop_26_quant_conv2d_4_post_activation_minIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27О
AssignVariableOp_27AssignVariableOp6assignvariableop_27_quant_conv2d_4_post_activation_maxIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Й
AssignVariableOp_28AssignVariableOp1assignvariableop_28_quant_conv2d_5_optimizer_stepIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Е
AssignVariableOp_29AssignVariableOp-assignvariableop_29_quant_conv2d_5_kernel_minIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Е
AssignVariableOp_30AssignVariableOp-assignvariableop_30_quant_conv2d_5_kernel_maxIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31О
AssignVariableOp_31AssignVariableOp6assignvariableop_31_quant_conv2d_5_post_activation_minIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32О
AssignVariableOp_32AssignVariableOp6assignvariableop_32_quant_conv2d_5_post_activation_maxIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33З
AssignVariableOp_33AssignVariableOp/assignvariableop_33_quant_lambda_optimizer_stepIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Й
AssignVariableOp_34AssignVariableOp1assignvariableop_34_quant_conv2d_6_optimizer_stepIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Е
AssignVariableOp_35AssignVariableOp-assignvariableop_35_quant_conv2d_6_kernel_minIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Е
AssignVariableOp_36AssignVariableOp-assignvariableop_36_quant_conv2d_6_kernel_maxIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37О
AssignVariableOp_37AssignVariableOp6assignvariableop_37_quant_conv2d_6_post_activation_minIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38О
AssignVariableOp_38AssignVariableOp6assignvariableop_38_quant_conv2d_6_post_activation_maxIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Д
AssignVariableOp_39AssignVariableOp,assignvariableop_39_quant_add_optimizer_stepIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40А
AssignVariableOp_40AssignVariableOp(assignvariableop_40_quant_add_output_minIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41А
AssignVariableOp_41AssignVariableOp(assignvariableop_41_quant_add_output_maxIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Й
AssignVariableOp_42AssignVariableOp1assignvariableop_42_quant_lambda_1_optimizer_stepIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Й
AssignVariableOp_43AssignVariableOp1assignvariableop_43_quant_lambda_2_optimizer_stepIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ђ
AssignVariableOp_44AssignVariableOpassignvariableop_44_beta_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ђ
AssignVariableOp_45AssignVariableOpassignvariableop_45_beta_2Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ё
AssignVariableOp_46AssignVariableOpassignvariableop_46_decayIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Љ
AssignVariableOp_47AssignVariableOp!assignvariableop_47_learning_rateIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_48Ѕ
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_iterIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Љ
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ї
AssignVariableOp_50AssignVariableOpassignvariableop_50_conv2d_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ћ
AssignVariableOp_51AssignVariableOp#assignvariableop_51_conv2d_1_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Љ
AssignVariableOp_52AssignVariableOp!assignvariableop_52_conv2d_1_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ћ
AssignVariableOp_53AssignVariableOp#assignvariableop_53_conv2d_2_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Љ
AssignVariableOp_54AssignVariableOp!assignvariableop_54_conv2d_2_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ћ
AssignVariableOp_55AssignVariableOp#assignvariableop_55_conv2d_3_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Љ
AssignVariableOp_56AssignVariableOp!assignvariableop_56_conv2d_3_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ћ
AssignVariableOp_57AssignVariableOp#assignvariableop_57_conv2d_4_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Љ
AssignVariableOp_58AssignVariableOp!assignvariableop_58_conv2d_4_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ћ
AssignVariableOp_59AssignVariableOp#assignvariableop_59_conv2d_5_kernelIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Љ
AssignVariableOp_60AssignVariableOp!assignvariableop_60_conv2d_5_biasIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ћ
AssignVariableOp_61AssignVariableOp#assignvariableop_61_conv2d_6_kernelIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Љ
AssignVariableOp_62AssignVariableOp!assignvariableop_62_conv2d_6_biasIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ё
AssignVariableOp_63AssignVariableOpassignvariableop_63_totalIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ё
AssignVariableOp_64AssignVariableOpassignvariableop_64_countIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65А
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_conv2d_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Ў
AssignVariableOp_66AssignVariableOp&assignvariableop_66_adam_conv2d_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67В
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68А
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69В
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70А
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71В
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_3_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72А
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_3_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73В
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv2d_4_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74А
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv2d_4_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75В
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv2d_5_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76А
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv2d_5_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77В
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv2d_6_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78А
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv2d_6_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79А
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_conv2d_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Ў
AssignVariableOp_80AssignVariableOp&assignvariableop_80_adam_conv2d_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81В
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv2d_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82А
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv2d_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83В
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv2d_2_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84А
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv2d_2_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85В
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv2d_3_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86А
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv2d_3_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87В
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv2d_4_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88А
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv2d_4_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89В
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv2d_5_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90А
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv2d_5_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91В
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_conv2d_6_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92А
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_conv2d_6_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_929
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpм
Identity_93Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_93Я
Identity_94IdentityIdentity_93:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92*
T0*
_output_shapes
: 2
Identity_94"#
identity_94Identity_94:output:0*б
_input_shapesП
М: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_92:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ц
з
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124174
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisс
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/3:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/4:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/5:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/6:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/7:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/8
№'
Ј
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2121125

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2121020

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ

F__inference_quant_add_layer_call_and_return_conditional_losses_2121200

inputs
inputs_1K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1q
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
addю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1У
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:09^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ю'
І
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2120950

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б

F__inference_quant_add_layer_call_and_return_conditional_losses_2124327
inputs_0
inputs_1K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1s
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
addю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1У
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:09^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
№'
Ј
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123661

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
дќ
ќ@
"__inference__wrapped_model_2120907
input_1`
Vmodel_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: u
[model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:k
]model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:k
]model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:@
2model_quant_conv2d_biasadd_readvariableop_resource:^
Tmodel_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: `
Vmodel_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_1_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_2_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_3_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_4_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_5_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: w
]model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:m
_model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:m
_model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:B
4model_quant_conv2d_6_biasadd_readvariableop_resource:`
Vmodel_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: [
Qmodel_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: ]
Smodel_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂHmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂJmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ)model/quant_conv2d/BiasAdd/ReadVariableOpЂRmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂTmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂTmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂKmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂMmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_1/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_2/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_3/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_4/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_5/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ+model/quant_conv2d_6/BiasAdd/ReadVariableOpЂTmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂVmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂVmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂMmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ЂMmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂOmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1­
Mmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
>model/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_1Umodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsЬ
Rmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp[model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02T
Rmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЦ
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp]model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ц
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp]model_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Cmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelZmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0\model/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0\model/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2E
Cmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЭ
model/quant_conv2d/Conv2DConv2DHmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Mmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d/Conv2DХ
)model/quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp2model_quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model/quant_conv2d/BiasAdd/ReadVariableOpц
model/quant_conv2d/BiasAddBiasAdd"model/quant_conv2d/Conv2D:output:01model/quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d/BiasAddЋ
model/quant_conv2d/ReluRelu#model/quant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d/ReluЇ
Kmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTmodel_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp­
Mmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVmodel_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1­
<model/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/quant_conv2d/Relu:activations:0Smodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Umodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2>
<model/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsв
Tmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelб
model/quant_conv2d_1/Conv2DConv2DFmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_1/Conv2DЫ
+model/quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_1/BiasAdd/ReadVariableOpю
model/quant_conv2d_1/BiasAddBiasAdd$model/quant_conv2d_1/Conv2D:output:03model/quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_1/BiasAddБ
model/quant_conv2d_1/ReluRelu%model/quant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_1/Relu­
Mmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_1_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1З
>model/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'model/quant_conv2d_1/Relu:activations:0Umodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsв
Tmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelг
model/quant_conv2d_2/Conv2DConv2DHmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_2/Conv2DЫ
+model/quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_2/BiasAdd/ReadVariableOpю
model/quant_conv2d_2/BiasAddBiasAdd$model/quant_conv2d_2/Conv2D:output:03model/quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_2/BiasAddБ
model/quant_conv2d_2/ReluRelu%model/quant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_2/Relu­
Mmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1З
>model/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'model/quant_conv2d_2/Relu:activations:0Umodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsв
Tmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelг
model/quant_conv2d_3/Conv2DConv2DHmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_3/Conv2DЫ
+model/quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_3/BiasAdd/ReadVariableOpю
model/quant_conv2d_3/BiasAddBiasAdd$model/quant_conv2d_3/Conv2D:output:03model/quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_3/BiasAddБ
model/quant_conv2d_3/ReluRelu%model/quant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_3/Relu­
Mmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_3_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1З
>model/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'model/quant_conv2d_3/Relu:activations:0Umodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsв
Tmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_4_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelг
model/quant_conv2d_4/Conv2DConv2DHmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_4/Conv2DЫ
+model/quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_4/BiasAdd/ReadVariableOpю
model/quant_conv2d_4/BiasAddBiasAdd$model/quant_conv2d_4/Conv2D:output:03model/quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_4/BiasAddБ
model/quant_conv2d_4/ReluRelu%model/quant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_4/Relu­
Mmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1З
>model/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'model/quant_conv2d_4/Relu:activations:0Umodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsв
Tmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_5_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelг
model/quant_conv2d_5/Conv2DConv2DHmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_5/Conv2DЫ
+model/quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_5/BiasAdd/ReadVariableOpю
model/quant_conv2d_5/BiasAddBiasAdd$model/quant_conv2d_5/Conv2D:output:03model/quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_5/BiasAddБ
model/quant_conv2d_5/ReluRelu%model/quant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_5/Relu­
Mmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1З
>model/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars'model/quant_conv2d_5/Relu:activations:0Umodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars
model/quant_lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
model/quant_lambda/concat/axisк
model/quant_lambda/concatConcatV2Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0'model/quant_lambda/concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_lambda/concatв
Tmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp]model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02V
Tmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЬ
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp_model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ь
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp_model_quant_conv2d_6_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02X
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2
Emodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannel\model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0^model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0^model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2G
Emodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelг
model/quant_conv2d_6/Conv2DConv2DHmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Omodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
model/quant_conv2d_6/Conv2DЫ
+model/quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4model_quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/quant_conv2d_6/BiasAdd/ReadVariableOpю
model/quant_conv2d_6/BiasAddBiasAdd$model/quant_conv2d_6/Conv2D:output:03model/quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_conv2d_6/BiasAdd­
Mmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpГ
Omodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_quant_conv2d_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Q
Omodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Е
>model/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/quant_conv2d_6/BiasAdd:output:0Umodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2@
>model/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsэ
model/quant_add/addAddV2"model/quant_lambda/concat:output:0Hmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
model/quant_add/add
Hmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpQmodel_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЄ
Jmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpSmodel_quant_add_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
9model/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsmodel/quant_add/add:z:0Pmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Rmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2;
9model/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsљ
!model/quant_lambda_1/DepthToSpaceDepthToSpaceCmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2#
!model/quant_lambda_1/DepthToSpaceЁ
,model/quant_lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2.
,model/quant_lambda_2/clip_by_value/Minimum/y
*model/quant_lambda_2/clip_by_value/MinimumMinimum*model/quant_lambda_1/DepthToSpace:output:05model/quant_lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2,
*model/quant_lambda_2/clip_by_value/Minimum
$model/quant_lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$model/quant_lambda_2/clip_by_value/yў
"model/quant_lambda_2/clip_by_valueMaximum.model/quant_lambda_2/clip_by_value/Minimum:z:0-model/quant_lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2$
"model/quant_lambda_2/clip_by_valueБ
IdentityIdentity&model/quant_lambda_2/clip_by_value:z:0I^model/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpK^model/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*^model/quant_conv2d/BiasAdd/ReadVariableOpS^model/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpU^model/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1U^model/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2L^model/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^model/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_1/BiasAdd/ReadVariableOpU^model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_2/BiasAdd/ReadVariableOpU^model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_3/BiasAdd/ReadVariableOpU^model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_4/BiasAdd/ReadVariableOpU^model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_5/BiasAdd/ReadVariableOpU^model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1,^model/quant_conv2d_6/BiasAdd/ReadVariableOpU^model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpW^model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1W^model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2N^model/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1N^model/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpP^model/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Hmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpHmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Jmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Jmodel/quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12V
)model/quant_conv2d/BiasAdd/ReadVariableOp)model/quant_conv2d/BiasAdd/ReadVariableOp2Ј
Rmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpRmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Ќ
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Ќ
Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Tmodel/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Kmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Mmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mmodel/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_1/BiasAdd/ReadVariableOp+model/quant_conv2d_1/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_2/BiasAdd/ReadVariableOp+model/quant_conv2d_2/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_3/BiasAdd/ReadVariableOp+model/quant_conv2d_3/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_4/BiasAdd/ReadVariableOp+model/quant_conv2d_4/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_5/BiasAdd/ReadVariableOp+model/quant_conv2d_5/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12Z
+model/quant_conv2d_6/BiasAdd/ReadVariableOp+model/quant_conv2d_6/BiasAdd/ReadVariableOp2Ќ
Tmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpTmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2А
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12А
Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Vmodel/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Mmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12
Mmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpMmodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2Ђ
Omodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1
ц
з
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124188
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisс
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:k g
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/2:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/3:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/4:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/5:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/6:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/7:kg
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/8
Є
кR
B__inference_model_layer_call_and_return_conditional_losses_2123294

inputsJ
@quantize_layer_allvaluesquantize_minimum_readvariableop_resource: J
@quantize_layer_allvaluesquantize_maximum_readvariableop_resource: V
<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource:@
2quant_conv2d_lastvaluequant_assignminlast_resource:@
2quant_conv2d_lastvaluequant_assignmaxlast_resource::
,quant_conv2d_biasadd_readvariableop_resource:M
Cquant_conv2d_movingavgquantize_assignminema_readvariableop_resource: M
Cquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_1_lastvaluequant_assignminlast_resource:B
4quant_conv2d_1_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_1_biasadd_readvariableop_resource:O
Equant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_2_lastvaluequant_assignminlast_resource:B
4quant_conv2d_2_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:O
Equant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_3_lastvaluequant_assignminlast_resource:B
4quant_conv2d_3_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_3_biasadd_readvariableop_resource:O
Equant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_4_lastvaluequant_assignminlast_resource:B
4quant_conv2d_4_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_4_biasadd_readvariableop_resource:O
Equant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_5_lastvaluequant_assignminlast_resource:B
4quant_conv2d_5_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_5_biasadd_readvariableop_resource:O
Equant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource: X
>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_6_lastvaluequant_assignminlast_resource:B
4quant_conv2d_6_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_6_biasadd_readvariableop_resource:O
Equant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource: J
@quant_add_movingavgquantize_assignminema_readvariableop_resource: J
@quant_add_movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂ<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂDquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ#quant_conv2d/BiasAdd/ReadVariableOpЂ)quant_conv2d/LastValueQuant/AssignMaxLastЂ)quant_conv2d/LastValueQuant/AssignMinLastЂ3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpЂ3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpЂLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂNquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂNquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂGquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_1/BiasAdd/ReadVariableOpЂ+quant_conv2d_1/LastValueQuant/AssignMaxLastЂ+quant_conv2d_1/LastValueQuant/AssignMinLastЂ5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_2/BiasAdd/ReadVariableOpЂ+quant_conv2d_2/LastValueQuant/AssignMaxLastЂ+quant_conv2d_2/LastValueQuant/AssignMinLastЂ5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_3/BiasAdd/ReadVariableOpЂ+quant_conv2d_3/LastValueQuant/AssignMaxLastЂ+quant_conv2d_3/LastValueQuant/AssignMinLastЂ5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_4/BiasAdd/ReadVariableOpЂ+quant_conv2d_4/LastValueQuant/AssignMaxLastЂ+quant_conv2d_4/LastValueQuant/AssignMinLastЂ5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_5/BiasAdd/ReadVariableOpЂ+quant_conv2d_5/LastValueQuant/AssignMaxLastЂ+quant_conv2d_5/LastValueQuant/AssignMinLastЂ5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ%quant_conv2d_6/BiasAdd/ReadVariableOpЂ+quant_conv2d_6/LastValueQuant/AssignMaxLastЂ+quant_conv2d_6/LastValueQuant/AssignMinLastЂ5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOpЂ5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpЂNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂPquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂPquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ЂAquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂAquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpЂGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ2quantize_layer/AllValuesQuantize/AssignMaxAllValueЂ2quantize_layer/AllValuesQuantize/AssignMinAllValueЂGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂIquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ђ7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpЂ7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpЉ
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quantize_layer/AllValuesQuantize/ConstЗ
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMin­
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quantize_layer/AllValuesQuantize/Const_1Й
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMaxы
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpѕ
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/MinimumЁ
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Minimum_1/yщ
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Minimum_1ы
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpѕ
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/MaximumЁ
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Maximum_1/yщ
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Maximum_1Ы
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMinAllValueЫ
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMaxAllValueР
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpФ
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ў
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsя
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype025
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpХ
6quant_conv2d/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          28
6quant_conv2d/LastValueQuant/BatchMin/reduction_indicesі
$quant_conv2d/LastValueQuant/BatchMinMin;quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2&
$quant_conv2d/LastValueQuant/BatchMinя
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype025
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpХ
6quant_conv2d/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          28
6quant_conv2d/LastValueQuant/BatchMax/reduction_indicesі
$quant_conv2d/LastValueQuant/BatchMaxMax;quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2&
$quant_conv2d/LastValueQuant/BatchMax
%quant_conv2d/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2'
%quant_conv2d/LastValueQuant/truediv/yй
#quant_conv2d/LastValueQuant/truedivRealDiv-quant_conv2d/LastValueQuant/BatchMax:output:0.quant_conv2d/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/truedivв
#quant_conv2d/LastValueQuant/MinimumMinimum-quant_conv2d/LastValueQuant/BatchMin:output:0'quant_conv2d/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/Minimum
!quant_conv2d/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2#
!quant_conv2d/LastValueQuant/mul/yЩ
quant_conv2d/LastValueQuant/mulMul-quant_conv2d/LastValueQuant/BatchMin:output:0*quant_conv2d/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2!
quant_conv2d/LastValueQuant/mulЮ
#quant_conv2d/LastValueQuant/MaximumMaximum-quant_conv2d/LastValueQuant/BatchMax:output:0#quant_conv2d/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/Maximumъ
)quant_conv2d/LastValueQuant/AssignMinLastAssignVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource'quant_conv2d/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02+
)quant_conv2d/LastValueQuant/AssignMinLastъ
)quant_conv2d/LastValueQuant/AssignMaxLastAssignVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource'quant_conv2d/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02+
)quant_conv2d/LastValueQuant/AssignMaxLastЁ
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02N
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЛ
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource*^quant_conv2d/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Л
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource*^quant_conv2d/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2х
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЕ
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d/Conv2DГ
#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#quant_conv2d/BiasAdd/ReadVariableOpЮ
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d/BiasAdd
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d/ReluЅ
$quant_conv2d/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2&
$quant_conv2d/MovingAvgQuantize/ConstЪ
'quant_conv2d/MovingAvgQuantize/BatchMinMinquant_conv2d/Relu:activations:0-quant_conv2d/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2)
'quant_conv2d/MovingAvgQuantize/BatchMinЉ
&quant_conv2d/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d/MovingAvgQuantize/Const_1Ь
'quant_conv2d/MovingAvgQuantize/BatchMaxMaxquant_conv2d/Relu:activations:0/quant_conv2d/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2)
'quant_conv2d/MovingAvgQuantize/BatchMax
(quant_conv2d/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(quant_conv2d/MovingAvgQuantize/Minimum/yс
&quant_conv2d/MovingAvgQuantize/MinimumMinimum0quant_conv2d/MovingAvgQuantize/BatchMin:output:01quant_conv2d/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2(
&quant_conv2d/MovingAvgQuantize/Minimum
(quant_conv2d/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(quant_conv2d/MovingAvgQuantize/Maximum/yс
&quant_conv2d/MovingAvgQuantize/MaximumMaximum0quant_conv2d/MovingAvgQuantize/BatchMax:output:01quant_conv2d/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2(
&quant_conv2d/MovingAvgQuantize/MaximumЋ
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:23
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decayє
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02<
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpњ
/quant_conv2d/MovingAvgQuantize/AssignMinEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMinEma/subћ
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMinEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mulѓ
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMinEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02A
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЋ
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:23
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decayє
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02<
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpњ
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/subћ
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMaxEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mulѓ
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMaxEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02A
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЬ
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02G
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpа
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ28
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsѕ
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_1/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_1/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_1/LastValueQuant/BatchMinMin=quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_1/LastValueQuant/BatchMinѕ
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_1/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_1/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_1/LastValueQuant/BatchMaxMax=quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_1/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_1/LastValueQuant/BatchMax
'quant_conv2d_1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_1/LastValueQuant/truediv/yс
%quant_conv2d_1/LastValueQuant/truedivRealDiv/quant_conv2d_1/LastValueQuant/BatchMax:output:00quant_conv2d_1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_1/LastValueQuant/truedivк
%quant_conv2d_1/LastValueQuant/MinimumMinimum/quant_conv2d_1/LastValueQuant/BatchMin:output:0)quant_conv2d_1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_1/LastValueQuant/Minimum
#quant_conv2d_1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_1/LastValueQuant/mul/yб
!quant_conv2d_1/LastValueQuant/mulMul/quant_conv2d_1/LastValueQuant/BatchMin:output:0,quant_conv2d_1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_1/LastValueQuant/mulж
%quant_conv2d_1/LastValueQuant/MaximumMaximum/quant_conv2d_1/LastValueQuant/BatchMax:output:0%quant_conv2d_1/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_1/LastValueQuant/Maximumђ
+quant_conv2d_1/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource)quant_conv2d_1/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_1/LastValueQuant/AssignMinLastђ
+quant_conv2d_1/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource)quant_conv2d_1/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_1/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource,^quant_conv2d_1/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource,^quant_conv2d_1/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЙ
quant_conv2d_1/Conv2DConv2D@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_1/Conv2DЙ
%quant_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_1/BiasAdd/ReadVariableOpж
quant_conv2d_1/BiasAddBiasAddquant_conv2d_1/Conv2D:output:0-quant_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_1/BiasAdd
quant_conv2d_1/ReluReluquant_conv2d_1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_1/ReluЉ
&quant_conv2d_1/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_1/MovingAvgQuantize/Constв
)quant_conv2d_1/MovingAvgQuantize/BatchMinMin!quant_conv2d_1/Relu:activations:0/quant_conv2d_1/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_1/MovingAvgQuantize/BatchMin­
(quant_conv2d_1/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_1/MovingAvgQuantize/Const_1д
)quant_conv2d_1/MovingAvgQuantize/BatchMaxMax!quant_conv2d_1/Relu:activations:01quant_conv2d_1/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_1/MovingAvgQuantize/BatchMax
*quant_conv2d_1/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_1/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_1/MovingAvgQuantize/MinimumMinimum2quant_conv2d_1/MovingAvgQuantize/BatchMin:output:03quant_conv2d_1/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_1/MovingAvgQuantize/Minimum
*quant_conv2d_1/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_1/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_1/MovingAvgQuantize/MaximumMaximum2quant_conv2d_1/MovingAvgQuantize/BatchMax:output:03quant_conv2d_1/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_1/MovingAvgQuantize/MaximumЏ
3quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_1/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_1_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_1_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_1/Relu:activations:0Oquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVarsѕ
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_2/LastValueQuant/BatchMinMin=quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_2/LastValueQuant/BatchMinѕ
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_2/LastValueQuant/BatchMaxMax=quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_2/LastValueQuant/BatchMax
'quant_conv2d_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_2/LastValueQuant/truediv/yс
%quant_conv2d_2/LastValueQuant/truedivRealDiv/quant_conv2d_2/LastValueQuant/BatchMax:output:00quant_conv2d_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/truedivк
%quant_conv2d_2/LastValueQuant/MinimumMinimum/quant_conv2d_2/LastValueQuant/BatchMin:output:0)quant_conv2d_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/Minimum
#quant_conv2d_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_2/LastValueQuant/mul/yб
!quant_conv2d_2/LastValueQuant/mulMul/quant_conv2d_2/LastValueQuant/BatchMin:output:0,quant_conv2d_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_2/LastValueQuant/mulж
%quant_conv2d_2/LastValueQuant/MaximumMaximum/quant_conv2d_2/LastValueQuant/BatchMax:output:0%quant_conv2d_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/Maximumђ
+quant_conv2d_2/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource)quant_conv2d_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_2/LastValueQuant/AssignMinLastђ
+quant_conv2d_2/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource)quant_conv2d_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_2/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource,^quant_conv2d_2/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource,^quant_conv2d_2/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_2/Conv2DConv2DBquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_2/Conv2DЙ
%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_2/BiasAdd/ReadVariableOpж
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_2/BiasAdd
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_2/ReluЉ
&quant_conv2d_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_2/MovingAvgQuantize/Constв
)quant_conv2d_2/MovingAvgQuantize/BatchMinMin!quant_conv2d_2/Relu:activations:0/quant_conv2d_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_2/MovingAvgQuantize/BatchMin­
(quant_conv2d_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_2/MovingAvgQuantize/Const_1д
)quant_conv2d_2/MovingAvgQuantize/BatchMaxMax!quant_conv2d_2/Relu:activations:01quant_conv2d_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_2/MovingAvgQuantize/BatchMax
*quant_conv2d_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_2/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_2/MovingAvgQuantize/MinimumMinimum2quant_conv2d_2/MovingAvgQuantize/BatchMin:output:03quant_conv2d_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_2/MovingAvgQuantize/Minimum
*quant_conv2d_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_2/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_2/MovingAvgQuantize/MaximumMaximum2quant_conv2d_2/MovingAvgQuantize/BatchMax:output:03quant_conv2d_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_2/MovingAvgQuantize/MaximumЏ
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsѕ
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_3/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_3/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_3/LastValueQuant/BatchMinMin=quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_3/LastValueQuant/BatchMinѕ
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_3/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_3/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_3/LastValueQuant/BatchMaxMax=quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_3/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_3/LastValueQuant/BatchMax
'quant_conv2d_3/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_3/LastValueQuant/truediv/yс
%quant_conv2d_3/LastValueQuant/truedivRealDiv/quant_conv2d_3/LastValueQuant/BatchMax:output:00quant_conv2d_3/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_3/LastValueQuant/truedivк
%quant_conv2d_3/LastValueQuant/MinimumMinimum/quant_conv2d_3/LastValueQuant/BatchMin:output:0)quant_conv2d_3/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_3/LastValueQuant/Minimum
#quant_conv2d_3/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_3/LastValueQuant/mul/yб
!quant_conv2d_3/LastValueQuant/mulMul/quant_conv2d_3/LastValueQuant/BatchMin:output:0,quant_conv2d_3/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_3/LastValueQuant/mulж
%quant_conv2d_3/LastValueQuant/MaximumMaximum/quant_conv2d_3/LastValueQuant/BatchMax:output:0%quant_conv2d_3/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_3/LastValueQuant/Maximumђ
+quant_conv2d_3/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource)quant_conv2d_3/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_3/LastValueQuant/AssignMinLastђ
+quant_conv2d_3/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource)quant_conv2d_3/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_3/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource,^quant_conv2d_3/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource,^quant_conv2d_3/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_3/Conv2DConv2DBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_3/Conv2DЙ
%quant_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_3/BiasAdd/ReadVariableOpж
quant_conv2d_3/BiasAddBiasAddquant_conv2d_3/Conv2D:output:0-quant_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_3/BiasAdd
quant_conv2d_3/ReluReluquant_conv2d_3/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_3/ReluЉ
&quant_conv2d_3/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_3/MovingAvgQuantize/Constв
)quant_conv2d_3/MovingAvgQuantize/BatchMinMin!quant_conv2d_3/Relu:activations:0/quant_conv2d_3/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_3/MovingAvgQuantize/BatchMin­
(quant_conv2d_3/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_3/MovingAvgQuantize/Const_1д
)quant_conv2d_3/MovingAvgQuantize/BatchMaxMax!quant_conv2d_3/Relu:activations:01quant_conv2d_3/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_3/MovingAvgQuantize/BatchMax
*quant_conv2d_3/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_3/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_3/MovingAvgQuantize/MinimumMinimum2quant_conv2d_3/MovingAvgQuantize/BatchMin:output:03quant_conv2d_3/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_3/MovingAvgQuantize/Minimum
*quant_conv2d_3/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_3/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_3/MovingAvgQuantize/MaximumMaximum2quant_conv2d_3/MovingAvgQuantize/BatchMax:output:03quant_conv2d_3/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_3/MovingAvgQuantize/MaximumЏ
3quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_3/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_3_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_3_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_3/Relu:activations:0Oquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVarsѕ
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_4/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_4/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_4/LastValueQuant/BatchMinMin=quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_4/LastValueQuant/BatchMinѕ
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_4/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_4/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_4/LastValueQuant/BatchMaxMax=quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_4/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_4/LastValueQuant/BatchMax
'quant_conv2d_4/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_4/LastValueQuant/truediv/yс
%quant_conv2d_4/LastValueQuant/truedivRealDiv/quant_conv2d_4/LastValueQuant/BatchMax:output:00quant_conv2d_4/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_4/LastValueQuant/truedivк
%quant_conv2d_4/LastValueQuant/MinimumMinimum/quant_conv2d_4/LastValueQuant/BatchMin:output:0)quant_conv2d_4/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_4/LastValueQuant/Minimum
#quant_conv2d_4/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_4/LastValueQuant/mul/yб
!quant_conv2d_4/LastValueQuant/mulMul/quant_conv2d_4/LastValueQuant/BatchMin:output:0,quant_conv2d_4/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_4/LastValueQuant/mulж
%quant_conv2d_4/LastValueQuant/MaximumMaximum/quant_conv2d_4/LastValueQuant/BatchMax:output:0%quant_conv2d_4/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_4/LastValueQuant/Maximumђ
+quant_conv2d_4/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource)quant_conv2d_4/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_4/LastValueQuant/AssignMinLastђ
+quant_conv2d_4/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource)quant_conv2d_4/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_4/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_4_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_4_lastvaluequant_assignminlast_resource,^quant_conv2d_4/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_4_lastvaluequant_assignmaxlast_resource,^quant_conv2d_4/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_4/Conv2DConv2DBquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_4/Conv2DЙ
%quant_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_4/BiasAdd/ReadVariableOpж
quant_conv2d_4/BiasAddBiasAddquant_conv2d_4/Conv2D:output:0-quant_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_4/BiasAdd
quant_conv2d_4/ReluReluquant_conv2d_4/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_4/ReluЉ
&quant_conv2d_4/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_4/MovingAvgQuantize/Constв
)quant_conv2d_4/MovingAvgQuantize/BatchMinMin!quant_conv2d_4/Relu:activations:0/quant_conv2d_4/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_4/MovingAvgQuantize/BatchMin­
(quant_conv2d_4/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_4/MovingAvgQuantize/Const_1д
)quant_conv2d_4/MovingAvgQuantize/BatchMaxMax!quant_conv2d_4/Relu:activations:01quant_conv2d_4/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_4/MovingAvgQuantize/BatchMax
*quant_conv2d_4/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_4/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_4/MovingAvgQuantize/MinimumMinimum2quant_conv2d_4/MovingAvgQuantize/BatchMin:output:03quant_conv2d_4/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_4/MovingAvgQuantize/Minimum
*quant_conv2d_4/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_4/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_4/MovingAvgQuantize/MaximumMaximum2quant_conv2d_4/MovingAvgQuantize/BatchMax:output:03quant_conv2d_4/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_4/MovingAvgQuantize/MaximumЏ
3quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_4/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_4_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_4_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_4/Relu:activations:0Oquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsѕ
5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_5/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_5/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_5/LastValueQuant/BatchMinMin=quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_5/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_5/LastValueQuant/BatchMinѕ
5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_5/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_5/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_5/LastValueQuant/BatchMaxMax=quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_5/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_5/LastValueQuant/BatchMax
'quant_conv2d_5/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_5/LastValueQuant/truediv/yс
%quant_conv2d_5/LastValueQuant/truedivRealDiv/quant_conv2d_5/LastValueQuant/BatchMax:output:00quant_conv2d_5/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_5/LastValueQuant/truedivк
%quant_conv2d_5/LastValueQuant/MinimumMinimum/quant_conv2d_5/LastValueQuant/BatchMin:output:0)quant_conv2d_5/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_5/LastValueQuant/Minimum
#quant_conv2d_5/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_5/LastValueQuant/mul/yб
!quant_conv2d_5/LastValueQuant/mulMul/quant_conv2d_5/LastValueQuant/BatchMin:output:0,quant_conv2d_5/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_5/LastValueQuant/mulж
%quant_conv2d_5/LastValueQuant/MaximumMaximum/quant_conv2d_5/LastValueQuant/BatchMax:output:0%quant_conv2d_5/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_5/LastValueQuant/Maximumђ
+quant_conv2d_5/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_5_lastvaluequant_assignminlast_resource)quant_conv2d_5/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_5/LastValueQuant/AssignMinLastђ
+quant_conv2d_5/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_5_lastvaluequant_assignmaxlast_resource)quant_conv2d_5/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_5/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_5_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_5_lastvaluequant_assignminlast_resource,^quant_conv2d_5/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_5_lastvaluequant_assignmaxlast_resource,^quant_conv2d_5/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_5/Conv2DConv2DBquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_5/Conv2DЙ
%quant_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_5/BiasAdd/ReadVariableOpж
quant_conv2d_5/BiasAddBiasAddquant_conv2d_5/Conv2D:output:0-quant_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_5/BiasAdd
quant_conv2d_5/ReluReluquant_conv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_5/ReluЉ
&quant_conv2d_5/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_5/MovingAvgQuantize/Constв
)quant_conv2d_5/MovingAvgQuantize/BatchMinMin!quant_conv2d_5/Relu:activations:0/quant_conv2d_5/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_5/MovingAvgQuantize/BatchMin­
(quant_conv2d_5/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_5/MovingAvgQuantize/Const_1д
)quant_conv2d_5/MovingAvgQuantize/BatchMaxMax!quant_conv2d_5/Relu:activations:01quant_conv2d_5/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_5/MovingAvgQuantize/BatchMax
*quant_conv2d_5/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_5/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_5/MovingAvgQuantize/MinimumMinimum2quant_conv2d_5/MovingAvgQuantize/BatchMin:output:03quant_conv2d_5/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_5/MovingAvgQuantize/Minimum
*quant_conv2d_5/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_5/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_5/MovingAvgQuantize/MaximumMaximum2quant_conv2d_5/MovingAvgQuantize/BatchMax:output:03quant_conv2d_5/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_5/MovingAvgQuantize/MaximumЏ
3quant_conv2d_5/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_5/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_5/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_5/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_5/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_5/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_5/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_5_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_5_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_5/Relu:activations:0Oquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsv
quant_lambda/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
quant_lambda/concat/axis
quant_lambda/concatConcatV2Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Bquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0!quant_lambda/concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_lambda/concatѕ
5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpЩ
8quant_conv2d_6/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_6/LastValueQuant/BatchMin/reduction_indicesў
&quant_conv2d_6/LastValueQuant/BatchMinMin=quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_6/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_6/LastValueQuant/BatchMinѕ
5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOpЩ
8quant_conv2d_6/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_6/LastValueQuant/BatchMax/reduction_indicesў
&quant_conv2d_6/LastValueQuant/BatchMaxMax=quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_6/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_6/LastValueQuant/BatchMax
'quant_conv2d_6/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2)
'quant_conv2d_6/LastValueQuant/truediv/yс
%quant_conv2d_6/LastValueQuant/truedivRealDiv/quant_conv2d_6/LastValueQuant/BatchMax:output:00quant_conv2d_6/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_6/LastValueQuant/truedivк
%quant_conv2d_6/LastValueQuant/MinimumMinimum/quant_conv2d_6/LastValueQuant/BatchMin:output:0)quant_conv2d_6/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_6/LastValueQuant/Minimum
#quant_conv2d_6/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2%
#quant_conv2d_6/LastValueQuant/mul/yб
!quant_conv2d_6/LastValueQuant/mulMul/quant_conv2d_6/LastValueQuant/BatchMin:output:0,quant_conv2d_6/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_6/LastValueQuant/mulж
%quant_conv2d_6/LastValueQuant/MaximumMaximum/quant_conv2d_6/LastValueQuant/BatchMax:output:0%quant_conv2d_6/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_6/LastValueQuant/Maximumђ
+quant_conv2d_6/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_6_lastvaluequant_assignminlast_resource)quant_conv2d_6/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_6/LastValueQuant/AssignMinLastђ
+quant_conv2d_6/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_6_lastvaluequant_assignmaxlast_resource)quant_conv2d_6/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_6/LastValueQuant/AssignMaxLastЇ
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_6_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpУ
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_6_lastvaluequant_assignminlast_resource,^quant_conv2d_6/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1У
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_6_lastvaluequant_assignmaxlast_resource,^quant_conv2d_6/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2я
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelЛ
quant_conv2d_6/Conv2DConv2DBquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
quant_conv2d_6/Conv2DЙ
%quant_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_6/BiasAdd/ReadVariableOpж
quant_conv2d_6/BiasAddBiasAddquant_conv2d_6/Conv2D:output:0-quant_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_conv2d_6/BiasAddЉ
&quant_conv2d_6/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_6/MovingAvgQuantize/Constа
)quant_conv2d_6/MovingAvgQuantize/BatchMinMinquant_conv2d_6/BiasAdd:output:0/quant_conv2d_6/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_6/MovingAvgQuantize/BatchMin­
(quant_conv2d_6/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_6/MovingAvgQuantize/Const_1в
)quant_conv2d_6/MovingAvgQuantize/BatchMaxMaxquant_conv2d_6/BiasAdd:output:01quant_conv2d_6/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_6/MovingAvgQuantize/BatchMax
*quant_conv2d_6/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_6/MovingAvgQuantize/Minimum/yщ
(quant_conv2d_6/MovingAvgQuantize/MinimumMinimum2quant_conv2d_6/MovingAvgQuantize/BatchMin:output:03quant_conv2d_6/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_6/MovingAvgQuantize/Minimum
*quant_conv2d_6/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_6/MovingAvgQuantize/Maximum/yщ
(quant_conv2d_6/MovingAvgQuantize/MaximumMaximum2quant_conv2d_6/MovingAvgQuantize/BatchMax:output:03quant_conv2d_6/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_6/MovingAvgQuantize/MaximumЏ
3quant_conv2d_6/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_6/MovingAvgQuantize/AssignMinEma/decayњ
<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_6/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/sub
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_6/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_6/MovingAvgQuantize/AssignMinEma/mul§
Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_6/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЏ
3quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/decayњ
<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_6/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/sub
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/mul§
Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpд
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_6_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpи
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_6_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d_6/BiasAdd:output:0Oquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2:
8quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsе
quant_add/addAddV2quant_lambda/concat:output:0Bquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_add/add
!quant_add/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!quant_add/MovingAvgQuantize/ConstГ
$quant_add/MovingAvgQuantize/BatchMinMinquant_add/add:z:0*quant_add/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2&
$quant_add/MovingAvgQuantize/BatchMinЃ
#quant_add/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#quant_add/MovingAvgQuantize/Const_1Е
$quant_add/MovingAvgQuantize/BatchMaxMaxquant_add/add:z:0,quant_add/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2&
$quant_add/MovingAvgQuantize/BatchMax
%quant_add/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%quant_add/MovingAvgQuantize/Minimum/yе
#quant_add/MovingAvgQuantize/MinimumMinimum-quant_add/MovingAvgQuantize/BatchMin:output:0.quant_add/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2%
#quant_add/MovingAvgQuantize/Minimum
%quant_add/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%quant_add/MovingAvgQuantize/Maximum/yе
#quant_add/MovingAvgQuantize/MaximumMaximum-quant_add/MovingAvgQuantize/BatchMax:output:0.quant_add/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2%
#quant_add/MovingAvgQuantize/MaximumЅ
.quant_add/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.quant_add/MovingAvgQuantize/AssignMinEma/decayы
7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype029
7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpю
,quant_add/MovingAvgQuantize/AssignMinEma/subSub?quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0'quant_add/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2.
,quant_add/MovingAvgQuantize/AssignMinEma/subя
,quant_add/MovingAvgQuantize/AssignMinEma/mulMul0quant_add/MovingAvgQuantize/AssignMinEma/sub:z:07quant_add/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2.
,quant_add/MovingAvgQuantize/AssignMinEma/mulф
<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource0quant_add/MovingAvgQuantize/AssignMinEma/mul:z:08^quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02>
<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЅ
.quant_add/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.quant_add/MovingAvgQuantize/AssignMaxEma/decayы
7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype029
7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOpю
,quant_add/MovingAvgQuantize/AssignMaxEma/subSub?quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0'quant_add/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2.
,quant_add/MovingAvgQuantize/AssignMaxEma/subя
,quant_add/MovingAvgQuantize/AssignMaxEma/mulMul0quant_add/MovingAvgQuantize/AssignMaxEma/sub:z:07quant_add/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2.
,quant_add/MovingAvgQuantize/AssignMaxEma/mulф
<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource0quant_add/MovingAvgQuantize/AssignMaxEma/mul:z:08^quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02>
<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpР
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quant_add_movingavgquantize_assignminema_readvariableop_resource=^quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02D
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpФ
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quant_add_movingavgquantize_assignmaxema_readvariableop_resource=^quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02F
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ѕ
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_add/add:z:0Jquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Lquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVarsч
quant_lambda_1/DepthToSpaceDepthToSpace=quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
quant_lambda_1/DepthToSpace
&quant_lambda_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2(
&quant_lambda_2/clip_by_value/Minimum/yњ
$quant_lambda_2/clip_by_value/MinimumMinimum$quant_lambda_1/DepthToSpace:output:0/quant_lambda_2/clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2&
$quant_lambda_2/clip_by_value/Minimum
quant_lambda_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
quant_lambda_2/clip_by_value/yц
quant_lambda_2/clip_by_valueMaximum(quant_lambda_2/clip_by_value/Minimum:z:0'quant_lambda_2/clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
quant_lambda_2/clip_by_value8
IdentityIdentity quant_lambda_2/clip_by_value:z:0=^quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp8^quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp=^quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp8^quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOpC^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpE^quant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1$^quant_conv2d/BiasAdd/ReadVariableOp*^quant_conv2d/LastValueQuant/AssignMaxLast*^quant_conv2d/LastValueQuant/AssignMinLast4^quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp4^quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpF^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_1/BiasAdd/ReadVariableOp,^quant_conv2d_1/LastValueQuant/AssignMaxLast,^quant_conv2d_1/LastValueQuant/AssignMinLast6^quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_2/BiasAdd/ReadVariableOp,^quant_conv2d_2/LastValueQuant/AssignMaxLast,^quant_conv2d_2/LastValueQuant/AssignMinLast6^quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_3/BiasAdd/ReadVariableOp,^quant_conv2d_3/LastValueQuant/AssignMaxLast,^quant_conv2d_3/LastValueQuant/AssignMinLast6^quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_4/BiasAdd/ReadVariableOp,^quant_conv2d_4/LastValueQuant/AssignMaxLast,^quant_conv2d_4/LastValueQuant/AssignMinLast6^quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_5/BiasAdd/ReadVariableOp,^quant_conv2d_5/LastValueQuant/AssignMaxLast,^quant_conv2d_5/LastValueQuant/AssignMinLast6^quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_conv2d_6/BiasAdd/ReadVariableOp,^quant_conv2d_6/LastValueQuant/AssignMaxLast,^quant_conv2d_6/LastValueQuant/AssignMinLast6^quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_13^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValueH^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp8^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<quant_add/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2r
7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp7quant_add/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2|
<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<quant_add/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2r
7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp7quant_add/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Bquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpBquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Dquant_add/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2V
)quant_conv2d/LastValueQuant/AssignMaxLast)quant_conv2d/LastValueQuant/AssignMaxLast2V
)quant_conv2d/LastValueQuant/AssignMinLast)quant_conv2d/LastValueQuant/AssignMinLast2j
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp2j
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp2
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12 
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_1/BiasAdd/ReadVariableOp%quant_conv2d_1/BiasAdd/ReadVariableOp2Z
+quant_conv2d_1/LastValueQuant/AssignMaxLast+quant_conv2d_1/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_1/LastValueQuant/AssignMinLast+quant_conv2d_1/LastValueQuant/AssignMinLast2n
5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_1/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_1/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_1/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2Z
+quant_conv2d_2/LastValueQuant/AssignMaxLast+quant_conv2d_2/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_2/LastValueQuant/AssignMinLast+quant_conv2d_2/LastValueQuant/AssignMinLast2n
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_3/BiasAdd/ReadVariableOp%quant_conv2d_3/BiasAdd/ReadVariableOp2Z
+quant_conv2d_3/LastValueQuant/AssignMaxLast+quant_conv2d_3/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_3/LastValueQuant/AssignMinLast+quant_conv2d_3/LastValueQuant/AssignMinLast2n
5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_3/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_3/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_3/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_4/BiasAdd/ReadVariableOp%quant_conv2d_4/BiasAdd/ReadVariableOp2Z
+quant_conv2d_4/LastValueQuant/AssignMaxLast+quant_conv2d_4/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_4/LastValueQuant/AssignMinLast+quant_conv2d_4/LastValueQuant/AssignMinLast2n
5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_4/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_5/BiasAdd/ReadVariableOp%quant_conv2d_5/BiasAdd/ReadVariableOp2Z
+quant_conv2d_5/LastValueQuant/AssignMaxLast+quant_conv2d_5/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_5/LastValueQuant/AssignMinLast+quant_conv2d_5/LastValueQuant/AssignMinLast2n
5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_5/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_5/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_conv2d_6/BiasAdd/ReadVariableOp%quant_conv2d_6/BiasAdd/ReadVariableOp2Z
+quant_conv2d_6/LastValueQuant/AssignMaxLast+quant_conv2d_6/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_6/LastValueQuant/AssignMinLast+quant_conv2d_6/LastValueQuant/AssignMinLast2n
5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_6/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_6/LastValueQuant/BatchMin/ReadVariableOp2 
Nquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2Є
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12Є
Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22
Aquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2
Aquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp2
Gquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue2
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp2r
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2121595

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123765

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
L
0__inference_quant_lambda_2_layer_call_fn_2124415

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21212212
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т­
)
 __inference__traced_save_2124722
file_prefix@
<savev2_quantize_layer_quantize_layer_min_read_readvariableop@
<savev2_quantize_layer_quantize_layer_max_read_readvariableop<
8savev2_quantize_layer_optimizer_step_read_readvariableop:
6savev2_quant_conv2d_optimizer_step_read_readvariableop6
2savev2_quant_conv2d_kernel_min_read_readvariableop6
2savev2_quant_conv2d_kernel_max_read_readvariableop?
;savev2_quant_conv2d_post_activation_min_read_readvariableop?
;savev2_quant_conv2d_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_1_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_1_kernel_min_read_readvariableop8
4savev2_quant_conv2d_1_kernel_max_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_1_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_2_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_2_kernel_min_read_readvariableop8
4savev2_quant_conv2d_2_kernel_max_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_3_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_3_kernel_min_read_readvariableop8
4savev2_quant_conv2d_3_kernel_max_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_3_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_4_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_4_kernel_min_read_readvariableop8
4savev2_quant_conv2d_4_kernel_max_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_4_post_activation_max_read_readvariableop<
8savev2_quant_conv2d_5_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_5_kernel_min_read_readvariableop8
4savev2_quant_conv2d_5_kernel_max_read_readvariableopA
=savev2_quant_conv2d_5_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_5_post_activation_max_read_readvariableop:
6savev2_quant_lambda_optimizer_step_read_readvariableop<
8savev2_quant_conv2d_6_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_6_kernel_min_read_readvariableop8
4savev2_quant_conv2d_6_kernel_max_read_readvariableopA
=savev2_quant_conv2d_6_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_6_post_activation_max_read_readvariableop7
3savev2_quant_add_optimizer_step_read_readvariableop3
/savev2_quant_add_output_min_read_readvariableop3
/savev2_quant_add_output_max_read_readvariableop<
8savev2_quant_lambda_1_optimizer_step_read_readvariableop<
8savev2_quant_lambda_2_optimizer_step_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЦ-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*и,
valueЮ,BЫ,^BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЧ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*б
valueЧBФ^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesФ'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_quantize_layer_quantize_layer_min_read_readvariableop<savev2_quantize_layer_quantize_layer_max_read_readvariableop8savev2_quantize_layer_optimizer_step_read_readvariableop6savev2_quant_conv2d_optimizer_step_read_readvariableop2savev2_quant_conv2d_kernel_min_read_readvariableop2savev2_quant_conv2d_kernel_max_read_readvariableop;savev2_quant_conv2d_post_activation_min_read_readvariableop;savev2_quant_conv2d_post_activation_max_read_readvariableop8savev2_quant_conv2d_1_optimizer_step_read_readvariableop4savev2_quant_conv2d_1_kernel_min_read_readvariableop4savev2_quant_conv2d_1_kernel_max_read_readvariableop=savev2_quant_conv2d_1_post_activation_min_read_readvariableop=savev2_quant_conv2d_1_post_activation_max_read_readvariableop8savev2_quant_conv2d_2_optimizer_step_read_readvariableop4savev2_quant_conv2d_2_kernel_min_read_readvariableop4savev2_quant_conv2d_2_kernel_max_read_readvariableop=savev2_quant_conv2d_2_post_activation_min_read_readvariableop=savev2_quant_conv2d_2_post_activation_max_read_readvariableop8savev2_quant_conv2d_3_optimizer_step_read_readvariableop4savev2_quant_conv2d_3_kernel_min_read_readvariableop4savev2_quant_conv2d_3_kernel_max_read_readvariableop=savev2_quant_conv2d_3_post_activation_min_read_readvariableop=savev2_quant_conv2d_3_post_activation_max_read_readvariableop8savev2_quant_conv2d_4_optimizer_step_read_readvariableop4savev2_quant_conv2d_4_kernel_min_read_readvariableop4savev2_quant_conv2d_4_kernel_max_read_readvariableop=savev2_quant_conv2d_4_post_activation_min_read_readvariableop=savev2_quant_conv2d_4_post_activation_max_read_readvariableop8savev2_quant_conv2d_5_optimizer_step_read_readvariableop4savev2_quant_conv2d_5_kernel_min_read_readvariableop4savev2_quant_conv2d_5_kernel_max_read_readvariableop=savev2_quant_conv2d_5_post_activation_min_read_readvariableop=savev2_quant_conv2d_5_post_activation_max_read_readvariableop6savev2_quant_lambda_optimizer_step_read_readvariableop8savev2_quant_conv2d_6_optimizer_step_read_readvariableop4savev2_quant_conv2d_6_kernel_min_read_readvariableop4savev2_quant_conv2d_6_kernel_max_read_readvariableop=savev2_quant_conv2d_6_post_activation_min_read_readvariableop=savev2_quant_conv2d_6_post_activation_max_read_readvariableop3savev2_quant_add_optimizer_step_read_readvariableop/savev2_quant_add_output_min_read_readvariableop/savev2_quant_add_output_max_read_readvariableop8savev2_quant_lambda_1_optimizer_step_read_readvariableop8savev2_quant_lambda_2_optimizer_step_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *l
dtypesb
`2^	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Џ
_input_shapes
: : : : : ::: : : ::: : : ::: : : ::: : : ::: : : ::: : : : ::: : : : : : : : : : : : ::::::::::::::: : ::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: : 


_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: : $

_output_shapes
:: %

_output_shapes
::&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::@

_output_shapes
: :A

_output_shapes
: :,B(
&
_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
:: O

_output_shapes
::,P(
&
_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:: S

_output_shapes
::,T(
&
_output_shapes
:: U

_output_shapes
::,V(
&
_output_shapes
:: W

_output_shapes
::,X(
&
_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
::,\(
&
_output_shapes
:: ]

_output_shapes
::^

_output_shapes
: 
ђ^
	
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2121847

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј]
	
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2121470

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ь
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
g
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124384

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
DepthToSpace
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
g
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2121335

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
clip_by_value/Minimum/yЏ
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yЊ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2120985

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѓ2
Й
F__inference_quant_add_layer_call_and_return_conditional_losses_2121395

inputs
inputs_1@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1q
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
add
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinadd:z:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxadd:z:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1У
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsadd:z:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsу
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123918

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
g
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124379

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
DepthToSpace
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
L
0__inference_quant_lambda_1_layer_call_fn_2124394

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21213512
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ&
Ј
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124234

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ь
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж	

0__inference_quant_conv2d_5_layer_call_fn_2124143

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21211252
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
T

B__inference_model_layer_call_and_return_conditional_losses_2122577
input_1 
quantize_layer_2122472:  
quantize_layer_2122474: .
quant_conv2d_2122477:"
quant_conv2d_2122479:"
quant_conv2d_2122481:"
quant_conv2d_2122483:
quant_conv2d_2122485: 
quant_conv2d_2122487: 0
quant_conv2d_1_2122490:$
quant_conv2d_1_2122492:$
quant_conv2d_1_2122494:$
quant_conv2d_1_2122496: 
quant_conv2d_1_2122498:  
quant_conv2d_1_2122500: 0
quant_conv2d_2_2122503:$
quant_conv2d_2_2122505:$
quant_conv2d_2_2122507:$
quant_conv2d_2_2122509: 
quant_conv2d_2_2122511:  
quant_conv2d_2_2122513: 0
quant_conv2d_3_2122516:$
quant_conv2d_3_2122518:$
quant_conv2d_3_2122520:$
quant_conv2d_3_2122522: 
quant_conv2d_3_2122524:  
quant_conv2d_3_2122526: 0
quant_conv2d_4_2122529:$
quant_conv2d_4_2122531:$
quant_conv2d_4_2122533:$
quant_conv2d_4_2122535: 
quant_conv2d_4_2122537:  
quant_conv2d_4_2122539: 0
quant_conv2d_5_2122542:$
quant_conv2d_5_2122544:$
quant_conv2d_5_2122546:$
quant_conv2d_5_2122548: 
quant_conv2d_5_2122550:  
quant_conv2d_5_2122552: 0
quant_conv2d_6_2122556:$
quant_conv2d_6_2122558:$
quant_conv2d_6_2122560:$
quant_conv2d_6_2122562: 
quant_conv2d_6_2122564:  
quant_conv2d_6_2122566: 
quant_add_2122569: 
quant_add_2122571: 
identityЂ!quant_add/StatefulPartitionedCallЂ$quant_conv2d/StatefulPartitionedCallЂ&quant_conv2d_1/StatefulPartitionedCallЂ&quant_conv2d_2/StatefulPartitionedCallЂ&quant_conv2d_3/StatefulPartitionedCallЂ&quant_conv2d_4/StatefulPartitionedCallЂ&quant_conv2d_5/StatefulPartitionedCallЂ&quant_conv2d_6/StatefulPartitionedCallЂ&quantize_layer/StatefulPartitionedCallе
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_2122472quantize_layer_2122474*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quantize_layer_layer_call_and_return_conditional_losses_21209232(
&quantize_layer/StatefulPartitionedCallг
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_2122477quant_conv2d_2122479quant_conv2d_2122481quant_conv2d_2122483quant_conv2d_2122485quant_conv2d_2122487*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_21209502&
$quant_conv2d/StatefulPartitionedCallу
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0quant_conv2d_1_2122490quant_conv2d_1_2122492quant_conv2d_1_2122494quant_conv2d_1_2122496quant_conv2d_1_2122498quant_conv2d_1_2122500*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21209852(
&quant_conv2d_1/StatefulPartitionedCallх
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0quant_conv2d_2_2122503quant_conv2d_2_2122505quant_conv2d_2_2122507quant_conv2d_2_2122509quant_conv2d_2_2122511quant_conv2d_2_2122513*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21210202(
&quant_conv2d_2/StatefulPartitionedCallх
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0quant_conv2d_3_2122516quant_conv2d_3_2122518quant_conv2d_3_2122520quant_conv2d_3_2122522quant_conv2d_3_2122524quant_conv2d_3_2122526*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_21210552(
&quant_conv2d_3/StatefulPartitionedCallх
&quant_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0quant_conv2d_4_2122529quant_conv2d_4_2122531quant_conv2d_4_2122533quant_conv2d_4_2122535quant_conv2d_4_2122537quant_conv2d_4_2122539*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_21210902(
&quant_conv2d_4/StatefulPartitionedCallх
&quant_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_4/StatefulPartitionedCall:output:0quant_conv2d_5_2122542quant_conv2d_5_2122544quant_conv2d_5_2122546quant_conv2d_5_2122548quant_conv2d_5_2122550quant_conv2d_5_2122552*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_21211252(
&quant_conv2d_5/StatefulPartitionedCallЙ
quant_lambda/PartitionedCallPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0/quantize_layer/StatefulPartitionedCall:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_quant_lambda_layer_call_and_return_conditional_losses_21211532
quant_lambda/PartitionedCallх
&quant_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_5/StatefulPartitionedCall:output:0quant_conv2d_6_2122556quant_conv2d_6_2122558quant_conv2d_6_2122560quant_conv2d_6_2122562quant_conv2d_6_2122564quant_conv2d_6_2122566*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_21211752(
&quant_conv2d_6/StatefulPartitionedCall
!quant_add/StatefulPartitionedCallStatefulPartitionedCall%quant_lambda/PartitionedCall:output:0/quant_conv2d_6/StatefulPartitionedCall:output:0quant_add_2122569quant_add_2122571*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_quant_add_layer_call_and_return_conditional_losses_21212002#
!quant_add/StatefulPartitionedCallЊ
quant_lambda_1/PartitionedCallPartitionedCall*quant_add/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_21212112 
quant_lambda_1/PartitionedCallЇ
quant_lambda_2/PartitionedCallPartitionedCall'quant_lambda_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21212212 
quant_lambda_2/PartitionedCallџ
IdentityIdentity'quant_lambda_2/PartitionedCall:output:0"^quant_add/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall'^quant_conv2d_4/StatefulPartitionedCall'^quant_conv2d_5/StatefulPartitionedCall'^quant_conv2d_6/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!quant_add/StatefulPartitionedCall!quant_add/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2P
&quant_conv2d_4/StatefulPartitionedCall&quant_conv2d_4/StatefulPartitionedCall2P
&quant_conv2d_5/StatefulPartitionedCall&quant_conv2d_5/StatefulPartitionedCall2P
&quant_conv2d_6/StatefulPartitionedCall&quant_conv2d_6/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:j f
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_1
И
L
0__inference_quant_lambda_2_layer_call_fn_2124420

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_21213352
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
g
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2121351

inputs
identity
DepthToSpaceDepthToSpaceinputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*

block_size2
DepthToSpace
IdentityIdentityDepthToSpace:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д
ў
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2120923

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂ8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ю
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Т
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)AllValuesQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:09^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2121055

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№^
	
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123606

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж	

0__inference_quant_conv2d_2_layer_call_fn_2123831

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_21210202
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
е
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2121523

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisп
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
concat}
IdentityIdentityconcat:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Њ
_input_shapes
:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2123973

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж	

0__inference_quant_conv2d_1_layer_call_fn_2123727

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21209852
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ^
	
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2121679

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№^
	
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2122015

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂLastValueQuant/AssignMaxLastЂLastValueQuant/AssignMinLastЂ&LastValueQuant/BatchMax/ReadVariableOpЂ&LastValueQuant/BatchMin/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЂ2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpЂ-MovingAvgQuantize/AssignMinEma/ReadVariableOpЂ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ш
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOpЋ
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesТ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMinШ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOpЋ
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesТ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/truediv/yЅ
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/MaximumЖ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastЖ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastњ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayЭ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subЧ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mulВ
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayЭ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpЦ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subЧ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mulВ
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsж
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№'
Ј
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2121090

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identityЂBiasAdd/ReadVariableOpЂ?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ЂALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Ђ8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpЂ:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Є
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelв
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluю
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpє
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ю
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsќ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
g
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124410

inputs
identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
clip_by_value/Minimum/yЏ
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yЊ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В	

0__inference_quant_conv2d_1_layer_call_fn_2123744

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_21219312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ЬL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultб
U
input_1J
serving_default_input_1:0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ\
quant_lambda_2J
StatefulPartitionedCall:0+џџџџџџџџџџџџџџџџџџџџџџџџџџџtensorflow/serving/predict:с
Э
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"иЧ
_tf_keras_networkЛЧ{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "name": "quantize_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d", "inbound_nodes": [[["quantize_layer", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_1", "inbound_nodes": [[["quant_conv2d", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_2", "inbound_nodes": [[["quant_conv2d_1", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_3", "inbound_nodes": [[["quant_conv2d_2", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_4", "inbound_nodes": [[["quant_conv2d_3", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_5", "inbound_nodes": [[["quant_conv2d_4", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAwAAACkB2gRh\neGlzKQLaAnRm2gZjb25jYXQpAdoGeF9saXN0qQByBgAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3Nw\nYWNlL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPg0AAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "name": "quant_lambda", "inbound_nodes": [[["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_6", "inbound_nodes": [[["quant_conv2d_5", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_add", "trainable": true, "dtype": "float32", "layer": {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "name": "quant_add", "inbound_nodes": [[["quant_lambda", 0, 0, {}], ["quant_conv2d_6", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAaACfACIAKECUwApAU4pA9oCdGbaAm5u\n2g5kZXB0aF90b19zcGFjZSkB2gF4KQHaBXNjYWxlqQB6OS9ob21lL2NjamlhaGFvL3dvcmtzcGFj\nZS9Nb2JpbGVTUi90cmlhbHMvYmFzZWxpbmUvYXJjaC5wedoIPGxhbWJkYT4aAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [3]}]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "name": "quant_lambda_1", "inbound_nodes": [[["quant_add", 0, 0, {}]]]}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMOAAAAdACgAXwAZAFkAqEDUwApA05nAAAAAAAAAADn\nAAAAAADgb0ApAtoBS9oEY2xpcCkB2gF4qQByBQAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3NwYWNl\nL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPhwAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "name": "quant_lambda_2", "inbound_nodes": [[["quant_lambda_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["quant_lambda_2", 0, 0]]}, "shared_object_id": 57, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "name": "quantize_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d", "inbound_nodes": [[["quantize_layer", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 9}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_1", "inbound_nodes": [[["quant_conv2d", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 15}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_2", "inbound_nodes": [[["quant_conv2d_1", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 21}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_3", "inbound_nodes": [[["quant_conv2d_2", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 27}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_4", "inbound_nodes": [[["quant_conv2d_3", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 33}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_5", "inbound_nodes": [[["quant_conv2d_4", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAwAAACkB2gRh\neGlzKQLaAnRm2gZjb25jYXQpAdoGeF9saXN0qQByBgAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3Nw\nYWNlL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPg0AAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 39}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 40}}, "name": "quant_lambda", "inbound_nodes": [[["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}]]], "shared_object_id": 41}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 42}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "name": "quant_conv2d_6", "inbound_nodes": [[["quant_conv2d_5", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_add", "trainable": true, "dtype": "float32", "layer": {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "shared_object_id": 48}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "name": "quant_add", "inbound_nodes": [[["quant_lambda", 0, 0, {}], ["quant_conv2d_6", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAaACfACIAKECUwApAU4pA9oCdGbaAm5u\n2g5kZXB0aF90b19zcGFjZSkB2gF4KQHaBXNjYWxlqQB6OS9ob21lL2NjamlhaGFvL3dvcmtzcGFj\nZS9Nb2JpbGVTUi90cmlhbHMvYmFzZWxpbmUvYXJjaC5wedoIPGxhbWJkYT4aAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [3]}]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 51}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 52}}, "name": "quant_lambda_1", "inbound_nodes": [[["quant_add", 0, 0, {}]]], "shared_object_id": 53}, {"class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMOAAAAdACgAXwAZAFkAqEDUwApA05nAAAAAAAAAADn\nAAAAAADgb0ApAtoBS9oEY2xpcCkB2gF4qQByBQAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3NwYWNl\nL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPhwAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 54}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 55}}, "name": "quant_lambda_2", "inbound_nodes": [[["quant_lambda_1", 0, 0, {}]]], "shared_object_id": 56}], "input_layers": [["input_1", 0, 0]], "output_layers": [["quant_lambda_2", 0, 0]]}}, "training_config": {"loss": "mae", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001250000059371814, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"ў
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Т
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"й
_tf_keras_layerП{"name": "quantize_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}

	layer
optimizer_step
_weight_vars

kernel_min
 
kernel_max
!_quantize_activations
"post_activation_min
#post_activation_max
$_output_quantizers
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+&call_and_return_all_conditional_losses
__call__"Ц
_tf_keras_layerЌ{"name": "quant_conv2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quantize_layer", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}

	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+&call_and_return_all_conditional_losses
__call__"Я
_tf_keras_layerЕ{"name": "quant_conv2d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 9}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d", 0, 0, {}]]], "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}

	6layer
7optimizer_step
8_weight_vars
9
kernel_min
:
kernel_max
;_quantize_activations
<post_activation_min
=post_activation_max
>_output_quantizers
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+&call_and_return_all_conditional_losses
__call__"в
_tf_keras_layerИ{"name": "quant_conv2d_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 15}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d_1", 0, 0, {}]]], "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}

	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+&call_and_return_all_conditional_losses
__call__"в
_tf_keras_layerИ{"name": "quant_conv2d_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 21}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d_2", 0, 0, {}]]], "shared_object_id": 26, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}

	Player
Qoptimizer_step
R_weight_vars
S
kernel_min
T
kernel_max
U_quantize_activations
Vpost_activation_min
Wpost_activation_max
X_output_quantizers
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+&call_and_return_all_conditional_losses
__call__"в
_tf_keras_layerИ{"name": "quant_conv2d_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 27}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d_3", 0, 0, {}]]], "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}

	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+&call_and_return_all_conditional_losses
__call__"в
_tf_keras_layerИ{"name": "quant_conv2d_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 33}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d_4", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
о
	jlayer
koptimizer_step
l_weight_vars
m_quantize_activations
n_output_quantizers
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+&call_and_return_all_conditional_losses
__call__"щ
_tf_keras_layerЯ{"name": "quant_lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAwAAACkB2gRh\neGlzKQLaAnRm2gZjb25jYXQpAdoGeF9saXN0qQByBgAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3Nw\nYWNlL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPg0AAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 39}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 40}}, "inbound_nodes": [[["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}], ["quantize_layer", 0, 0, {}]]], "shared_object_id": 41, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}, {"class_name": "TensorShape", "items": [null, null, null, 3]}]}

	slayer
toptimizer_step
u_weight_vars
v
kernel_min
w
kernel_max
x_quantize_activations
ypost_activation_min
zpost_activation_max
{_output_quantizers
|	variables
}regularization_losses
~trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"д
_tf_keras_layerК{"name": "quant_conv2d_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_conv2d_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 42}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "inbound_nodes": [[["quant_conv2d_5", 0, 0, {}]]], "shared_object_id": 47, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 27]}}
і

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Й
_tf_keras_layer{"name": "quant_add", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_add", "trainable": true, "dtype": "float32", "layer": {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "shared_object_id": 48}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "inbound_nodes": [[["quant_lambda", 0, 0, {}], ["quant_conv2d_6", 0, 0, {}]]], "shared_object_id": 50, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 27]}, {"class_name": "TensorShape", "items": [null, null, null, 27]}]}
 

layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ђ	
_tf_keras_layer	{"name": "quant_lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAaACfACIAKECUwApAU4pA9oCdGbaAm5u\n2g5kZXB0aF90b19zcGFjZSkB2gF4KQHaBXNjYWxlqQB6OS9ob21lL2NjamlhaGFvL3dvcmtzcGFj\nZS9Nb2JpbGVTUi90cmlhbHMvYmFzZWxpbmUvYXJjaC5wedoIPGxhbWJkYT4aAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [3]}]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 51}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 52}}, "inbound_nodes": [[["quant_add", 0, 0, {}]]], "shared_object_id": 53, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 27]}}
џ


layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"	
_tf_keras_layerч{"name": "quant_lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapperV2", "config": {"name": "quant_lambda_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMOAAAAdACgAXwAZAFkAqEDUwApA05nAAAAAAAAAADn\nAAAAAADgb0ApAtoBS9oEY2xpcCkB2gF4qQByBQAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3NwYWNl\nL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPhwAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 54}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}, "shared_object_id": 55}}, "inbound_nodes": [[["quant_lambda_1", 0, 0, {}]]], "shared_object_id": 56, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}

beta_1
beta_2

 decay
Ёlearning_rate
	Ђiter	Ѓmш	Єmщ	Ѕmъ	Іmы	Їmь	Јmэ	Љmю	Њmя	Ћm№	Ќmё	­mђ	Ўmѓ	Џmє	Аmѕ	Ѓvі	Єvї	Ѕvј	Іvљ	Їvњ	Јvћ	Љvќ	Њv§	Ћvў	Ќvџ	­v	Ўv	Џv	Аv"
	optimizer
љ
0
1
2
Ѓ3
Є4
5
6
 7
"8
#9
Ѕ10
І11
*12
,13
-14
/15
016
Ї17
Ј18
719
920
:21
<22
=23
Љ24
Њ25
D26
F27
G28
I29
J30
Ћ31
Ќ32
Q33
S34
T35
V36
W37
­38
Ў39
^40
`41
a42
c43
d44
k45
Џ46
А47
t48
v49
w50
y51
z52
53
54
55
56
57"
trackable_list_wrapper
 "
trackable_list_wrapper

Ѓ0
Є1
Ѕ2
І3
Ї4
Ј5
Љ6
Њ7
Ћ8
Ќ9
­10
Ў11
Џ12
А13"
trackable_list_wrapper
г
	variables
 Бlayer_regularization_losses
regularization_losses
Вnon_trainable_variables
trainable_variables
Гlayer_metrics
Дmetrics
Еlayers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
	variables
 Жlayer_regularization_losses
regularization_losses
Зnon_trainable_variables
trainable_variables
Иlayer_metrics
Йmetrics
Кlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
А
Ѓkernel
	Єbias
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+ &call_and_return_all_conditional_losses
Ё__call__"

_tf_keras_layerщ	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
#:! 2quant_conv2d/optimizer_step
(
П0"
trackable_list_wrapper
#:!2quant_conv2d/kernel_min
#:!2quant_conv2d/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_conv2d/post_activation_min
(:& 2 quant_conv2d/post_activation_max
 "
trackable_list_wrapper
S
Ѓ0
Є1
2
3
 4
"5
#6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
Е
%	variables
 Рlayer_regularization_losses
&regularization_losses
Сnon_trainable_variables
'trainable_variables
Тlayer_metrics
Уmetrics
Фlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Й
Ѕkernel
	Іbias
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
+Ђ&call_and_return_all_conditional_losses
Ѓ__call__"

_tf_keras_layerђ	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 9}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 28}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
%:# 2quant_conv2d_1/optimizer_step
(
Щ0"
trackable_list_wrapper
%:#2quant_conv2d_1/kernel_min
%:#2quant_conv2d_1/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_1/post_activation_min
*:( 2"quant_conv2d_1/post_activation_max
 "
trackable_list_wrapper
S
Ѕ0
І1
*2
,3
-4
/5
06"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
Е
2	variables
 Ъlayer_regularization_losses
3regularization_losses
Ыnon_trainable_variables
4trainable_variables
Ьlayer_metrics
Эmetrics
Юlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К
Їkernel
	Јbias
Я	variables
аregularization_losses
бtrainable_variables
в	keras_api
+Є&call_and_return_all_conditional_losses
Ѕ__call__"

_tf_keras_layerѓ	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 15}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 28}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
%:# 2quant_conv2d_2/optimizer_step
(
г0"
trackable_list_wrapper
%:#2quant_conv2d_2/kernel_min
%:#2quant_conv2d_2/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_2/post_activation_min
*:( 2"quant_conv2d_2/post_activation_max
 "
trackable_list_wrapper
S
Ї0
Ј1
72
93
:4
<5
=6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
Е
?	variables
 дlayer_regularization_losses
@regularization_losses
еnon_trainable_variables
Atrainable_variables
жlayer_metrics
зmetrics
иlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К
Љkernel
	Њbias
й	variables
кregularization_losses
лtrainable_variables
м	keras_api
+І&call_and_return_all_conditional_losses
Ї__call__"

_tf_keras_layerѓ	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 21}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 28}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
%:# 2quant_conv2d_3/optimizer_step
(
н0"
trackable_list_wrapper
%:#2quant_conv2d_3/kernel_min
%:#2quant_conv2d_3/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_3/post_activation_min
*:( 2"quant_conv2d_3/post_activation_max
 "
trackable_list_wrapper
S
Љ0
Њ1
D2
F3
G4
I5
J6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Љ0
Њ1"
trackable_list_wrapper
Е
L	variables
 оlayer_regularization_losses
Mregularization_losses
пnon_trainable_variables
Ntrainable_variables
рlayer_metrics
сmetrics
тlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К
Ћkernel
	Ќbias
у	variables
фregularization_losses
хtrainable_variables
ц	keras_api
+Ј&call_and_return_all_conditional_losses
Љ__call__"

_tf_keras_layerѓ	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 27}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 28}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
%:# 2quant_conv2d_4/optimizer_step
(
ч0"
trackable_list_wrapper
%:#2quant_conv2d_4/kernel_min
%:#2quant_conv2d_4/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_4/post_activation_min
*:( 2"quant_conv2d_4/post_activation_max
 "
trackable_list_wrapper
S
Ћ0
Ќ1
Q2
S3
T4
V5
W6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
Е
Y	variables
 шlayer_regularization_losses
Zregularization_losses
щnon_trainable_variables
[trainable_variables
ъlayer_metrics
ыmetrics
ьlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К
­kernel
	Ўbias
э	variables
юregularization_losses
яtrainable_variables
№	keras_api
+Њ&call_and_return_all_conditional_losses
Ћ__call__"

_tf_keras_layerѓ	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 33}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 28}}, "shared_object_id": 82}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 28]}}
%:# 2quant_conv2d_5/optimizer_step
(
ё0"
trackable_list_wrapper
%:#2quant_conv2d_5/kernel_min
%:#2quant_conv2d_5/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_5/post_activation_min
*:( 2"quant_conv2d_5/post_activation_max
 "
trackable_list_wrapper
S
­0
Ў1
^2
`3
a4
c5
d6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
­0
Ў1"
trackable_list_wrapper
Е
f	variables
 ђlayer_regularization_losses
gregularization_losses
ѓnon_trainable_variables
htrainable_variables
єlayer_metrics
ѕmetrics
іlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object

ї	variables
јregularization_losses
љtrainable_variables
њ	keras_api
+Ќ&call_and_return_all_conditional_losses
­__call__"
_tf_keras_layerш{"name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAwAAACkB2gRh\neGlzKQLaAnRm2gZjb25jYXQpAdoGeF9saXN0qQByBgAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3Nw\nYWNlL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPg0AAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 39}
#:! 2quant_lambda/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
o	variables
 ћlayer_regularization_losses
pregularization_losses
ќnon_trainable_variables
qtrainable_variables
§layer_metrics
ўmetrics
џlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
М
Џkernel
	Аbias
	variables
regularization_losses
trainable_variables
	keras_api
+Ў&call_and_return_all_conditional_losses
Џ__call__"

_tf_keras_layerѕ	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 27, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 42}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 27}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 27]}}
%:# 2quant_conv2d_6/optimizer_step
(
0"
trackable_list_wrapper
%:#2quant_conv2d_6/kernel_min
%:#2quant_conv2d_6/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_6/post_activation_min
*:( 2"quant_conv2d_6/post_activation_max
 "
trackable_list_wrapper
S
Џ0
А1
t2
v3
w4
y5
z6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Џ0
А1"
trackable_list_wrapper
Е
|	variables
 layer_regularization_losses
}regularization_losses
non_trainable_variables
~trainable_variables
layer_metrics
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
л
	variables
regularization_losses
trainable_variables
	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"Ц
_tf_keras_layerЌ{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "shared_object_id": 48, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, null, 27]}, {"class_name": "TensorShape", "items": [null, null, null, 27]}]}
 : 2quant_add/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2quant_add/output_min
: 2quant_add/output_max
<
min_var
max_var"
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 layer_regularization_losses
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
М
	variables
regularization_losses
trainable_variables
	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"Ї
_tf_keras_layer{"name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAaACfACIAKECUwApAU4pA9oCdGbaAm5u\n2g5kZXB0aF90b19zcGFjZSkB2gF4KQHaBXNjYWxlqQB6OS9ob21lL2NjamlhaGFvL3dvcmtzcGFj\nZS9Nb2JpbGVTUi90cmlhbHMvYmFzZWxpbmUvYXJjaC5wedoIPGxhbWJkYT4aAAAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [3]}]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 51}
%:# 2quant_lambda_1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 layer_regularization_losses
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object

	variables
regularization_losses
trainable_variables
	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layerш{"name": "lambda_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMOAAAAdACgAXwAZAFkAqEDUwApA05nAAAAAAAAAADn\nAAAAAADgb0ApAtoBS9oEY2xpcCkB2gF4qQByBQAAAHo5L2hvbWUvY2NqaWFoYW8vd29ya3NwYWNl\nL01vYmlsZVNSL3RyaWFscy9iYXNlbGluZS9hcmNoLnB52gg8bGFtYmRhPhwAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "trials.baseline.arch", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 54}
%:# 2quant_lambda_2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
  layer_regularization_losses
regularization_losses
Ёnon_trainable_variables
trainable_variables
Ђlayer_metrics
Ѓmetrics
Єlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
):'2conv2d_2/kernel
:2conv2d_2/bias
):'2conv2d_3/kernel
:2conv2d_3/bias
):'2conv2d_4/kernel
:2conv2d_4/bias
):'2conv2d_5/kernel
:2conv2d_5/bias
):'2conv2d_6/kernel
:2conv2d_6/bias
 "
trackable_list_wrapper
ћ
0
1
2
3
4
 5
"6
#7
*8
,9
-10
/11
012
713
914
:15
<16
=17
D18
F19
G20
I21
J22
Q23
S24
T25
V26
W27
^28
`29
a30
c31
d32
k33
t34
v35
w36
y37
z38
39
40
41
42
43"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
Ѕ0"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Є0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Є0"
trackable_list_wrapper
И
Л	variables
 Іlayer_regularization_losses
Мregularization_losses
Їnon_trainable_variables
Нtrainable_variables
Јlayer_metrics
Љmetrics
Њlayers
Ё__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
1
Ѓ0
Ћ2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
0
1
 2
"3
#4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
І0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
І0"
trackable_list_wrapper
И
Х	variables
 Ќlayer_regularization_losses
Цregularization_losses
­non_trainable_variables
Чtrainable_variables
Ўlayer_metrics
Џmetrics
Аlayers
Ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
1
Ѕ0
Б2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
*0
,1
-2
/3
04"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
(
Ј0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ј0"
trackable_list_wrapper
И
Я	variables
 Вlayer_regularization_losses
аregularization_losses
Гnon_trainable_variables
бtrainable_variables
Дlayer_metrics
Еmetrics
Жlayers
Ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
1
Ї0
З2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
70
91
:2
<3
=4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
(
Њ0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Њ0"
trackable_list_wrapper
И
й	variables
 Иlayer_regularization_losses
кregularization_losses
Йnon_trainable_variables
лtrainable_variables
Кlayer_metrics
Лmetrics
Мlayers
Ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
1
Љ0
Н2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
D0
F1
G2
I3
J4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
(
Ќ0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ќ0"
trackable_list_wrapper
И
у	variables
 Оlayer_regularization_losses
фregularization_losses
Пnon_trainable_variables
хtrainable_variables
Рlayer_metrics
Сmetrics
Тlayers
Љ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
1
Ћ0
У2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
Q0
S1
T2
V3
W4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
(
Ў0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ў0"
trackable_list_wrapper
И
э	variables
 Фlayer_regularization_losses
юregularization_losses
Хnon_trainable_variables
яtrainable_variables
Цlayer_metrics
Чmetrics
Шlayers
Ћ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
1
­0
Щ2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
^0
`1
a2
c3
d4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ї	variables
 Ъlayer_regularization_losses
јregularization_losses
Ыnon_trainable_variables
љtrainable_variables
Ьlayer_metrics
Эmetrics
Юlayers
­__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
j0"
trackable_list_wrapper
(
А0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
А0"
trackable_list_wrapper
И
	variables
 Яlayer_regularization_losses
regularization_losses
аnon_trainable_variables
trainable_variables
бlayer_metrics
вmetrics
гlayers
Џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
1
Џ0
д2"
trackable_tuple_wrapper
 "
trackable_list_wrapper
C
t0
v1
w2
y3
z4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 еlayer_regularization_losses
regularization_losses
жnon_trainable_variables
trainable_variables
зlayer_metrics
иmetrics
йlayers
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 кlayer_regularization_losses
regularization_losses
лnon_trainable_variables
trainable_variables
мlayer_metrics
нmetrics
оlayers
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 пlayer_regularization_losses
regularization_losses
рnon_trainable_variables
trainable_variables
сlayer_metrics
тmetrics
уlayers
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
и

фtotal

хcount
ц	variables
ч	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 84}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
min_var
 max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
,min_var
-max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
9min_var
:max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
Fmin_var
Gmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
Smin_var
Tmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
`min_var
amax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
vmin_var
wmax_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
ф0
х1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
ж2г
B__inference_model_layer_call_and_return_conditional_losses_2122930
B__inference_model_layer_call_and_return_conditional_losses_2123294
B__inference_model_layer_call_and_return_conditional_losses_2122577
B__inference_model_layer_call_and_return_conditional_losses_2122685Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
'__inference_model_layer_call_fn_2121319
'__inference_model_layer_call_fn_2123391
'__inference_model_layer_call_fn_2123488
'__inference_model_layer_call_fn_2122469Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
"__inference__wrapped_model_2120907а
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123497
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123518Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
0__inference_quantize_layer_layer_call_fn_2123527
0__inference_quantize_layer_layer_call_fn_2123536Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123557
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123606К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
.__inference_quant_conv2d_layer_call_fn_2123623
.__inference_quant_conv2d_layer_call_fn_2123640К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123661
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123710К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_1_layer_call_fn_2123727
0__inference_quant_conv2d_1_layer_call_fn_2123744К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123765
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123814К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_2_layer_call_fn_2123831
0__inference_quant_conv2d_2_layer_call_fn_2123848К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123869
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123918К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_3_layer_call_fn_2123935
0__inference_quant_conv2d_3_layer_call_fn_2123952К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2123973
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2124022К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_4_layer_call_fn_2124039
0__inference_quant_conv2d_4_layer_call_fn_2124056К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124077
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124126К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_5_layer_call_fn_2124143
0__inference_quant_conv2d_5_layer_call_fn_2124160К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124174
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124188К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
.__inference_quant_lambda_layer_call_fn_2124201
.__inference_quant_lambda_layer_call_fn_2124214К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124234
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124282К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_conv2d_6_layer_call_fn_2124299
0__inference_quant_conv2d_6_layer_call_fn_2124316К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
F__inference_quant_add_layer_call_and_return_conditional_losses_2124327
F__inference_quant_add_layer_call_and_return_conditional_losses_2124354К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
+__inference_quant_add_layer_call_fn_2124364
+__inference_quant_add_layer_call_fn_2124374К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124379
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124384К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_lambda_1_layer_call_fn_2124389
0__inference_quant_lambda_1_layer_call_fn_2124394К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124402
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124410К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
0__inference_quant_lambda_2_layer_call_fn_2124415
0__inference_quant_lambda_2_layer_call_fn_2124420К
БВ­
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЬBЩ
%__inference_signature_wrapper_2122790input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ц2УР
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
"__inference__wrapped_model_2120907ч>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzJЂG
@Ђ=
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "YЊV
T
quant_lambda_2B?
quant_lambda_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
B__inference_model_layer_call_and_return_conditional_losses_2122577е>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
B__inference_model_layer_call_and_return_conditional_losses_2122685е>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
B__inference_model_layer_call_and_return_conditional_losses_2122930д>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
B__inference_model_layer_call_and_return_conditional_losses_2123294д>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 є
'__inference_model_layer_call_fn_2121319Ш>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџє
'__inference_model_layer_call_fn_2122469Ш>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzRЂO
HЂE
;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
'__inference_model_layer_call_fn_2123391Ч>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџѓ
'__inference_model_layer_call_fn_2123488Ч>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzQЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЊ
F__inference_quant_add_layer_call_and_return_conditional_losses_2124327пЂ
Ђ
|
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Њ
F__inference_quant_add_layer_call_and_return_conditional_losses_2124354пЂ
Ђ
|
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
+__inference_quant_add_layer_call_fn_2124364вЂ
Ђ
|
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
+__inference_quant_add_layer_call_fn_2124374вЂ
Ђ
|
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123661Ѕ,-І/0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2123710Ѕ,-І/0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_1_layer_call_fn_2123727Ѕ,-І/0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_1_layer_call_fn_2123744Ѕ,-І/0MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123765Ї9:Ј<=MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2123814Ї9:Ј<=MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_2_layer_call_fn_2123831Ї9:Ј<=MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_2_layer_call_fn_2123848Ї9:Ј<=MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123869ЉFGЊIJMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2123918ЉFGЊIJMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_3_layer_call_fn_2123935ЉFGЊIJMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_3_layer_call_fn_2123952ЉFGЊIJMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2123973ЋSTЌVWMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_4_layer_call_and_return_conditional_losses_2124022ЋSTЌVWMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_4_layer_call_fn_2124039ЋSTЌVWMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_4_layer_call_fn_2124056ЋSTЌVWMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124077­`aЎcdMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_5_layer_call_and_return_conditional_losses_2124126­`aЎcdMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_5_layer_call_fn_2124143­`aЎcdMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_5_layer_call_fn_2124160­`aЎcdMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџъ
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124234ЏvwАyzMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ъ
K__inference_quant_conv2d_6_layer_call_and_return_conditional_losses_2124282ЏvwАyzMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
0__inference_quant_conv2d_6_layer_call_fn_2124299ЏvwАyzMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџТ
0__inference_quant_conv2d_6_layer_call_fn_2124316ЏvwАyzMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџш
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123557Ѓ Є"#MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ш
I__inference_quant_conv2d_layer_call_and_return_conditional_losses_2123606Ѓ Є"#MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Р
.__inference_quant_conv2d_layer_call_fn_2123623Ѓ Є"#MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџР
.__inference_quant_conv2d_layer_call_fn_2123640Ѓ Є"#MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџр
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124379MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 р
K__inference_quant_lambda_1_layer_call_and_return_conditional_losses_2124384MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
0__inference_quant_lambda_1_layer_call_fn_2124389MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџИ
0__inference_quant_lambda_1_layer_call_fn_2124394MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџр
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124402MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 р
K__inference_quant_lambda_2_layer_call_and_return_conditional_losses_2124410MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
0__inference_quant_lambda_2_layer_call_fn_2124415MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџИ
0__inference_quant_lambda_2_layer_call_fn_2124420MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџл
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124174ЩЂХ
НЂЙ
ВЎ
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/3+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/4+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/5+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/6+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/7+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/8+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 л
I__inference_quant_lambda_layer_call_and_return_conditional_losses_2124188ЩЂХ
НЂЙ
ВЎ
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/3+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/4+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/5+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/6+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/7+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/8+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Г
.__inference_quant_lambda_layer_call_fn_2124201ЩЂХ
НЂЙ
ВЎ
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/3+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/4+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/5+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/6+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/7+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/8+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџГ
.__inference_quant_lambda_layer_call_fn_2124214ЩЂХ
НЂЙ
ВЎ
<9
inputs/0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/3+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/4+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/5+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/6+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/7+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
<9
inputs/8+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџф
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123497MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
K__inference_quantize_layer_layer_call_and_return_conditional_losses_2123518MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
0__inference_quantize_layer_layer_call_fn_2123527MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
0__inference_quantize_layer_layer_call_fn_2123536MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
%__inference_signature_wrapper_2122790ђ>Ѓ Є"#Ѕ,-І/0Ї9:Ј<=ЉFGЊIJЋSTЌVW­`aЎcdЏvwАyzUЂR
Ђ 
KЊH
F
input_1;8
input_1+џџџџџџџџџџџџџџџџџџџџџџџџџџџ"YЊV
T
quant_lambda_2B?
quant_lambda_2+џџџџџџџџџџџџџџџџџџџџџџџџџџџ