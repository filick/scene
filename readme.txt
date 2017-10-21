For testing, remember to alter the file name when finishing a run to keep the best results.
Scores below is used to verify the code, only for reference.

test1    
resnet18_places_submit1_val.json
resnet18_places_submit1_test.json
0.5544943820224719

test2_* (average before softmax, actually model output is logits)
json marked by submit2
0.5726123595505618

test3_* (average after softmax, more common)
json marked by submit3
0.5728932584269663

test4_fcn (following resnet paper, fcn-style testing, except for a center_crop after Resize)
json marked by submit4
0.5589887640449438, a very poor setting due to no enough GPU

test5_ensemble (integrate outputs from test1,test2,test 3,test4)
json marked by submit5
0.5544 + 0.5589 -> 0.5631
