## Antimicrobial Peptide Identification Using Multi-scale Convolutional Network<br>
### Running Environment<br>
* Linux environment, Python 3<br>
* The two packages need to be available: numpy, keras<br>
* the fasta file must follow this format:
```
>unique_sequence_ID
fasta string
```
### proposed model<br>
* The proposed_model.py can be operated in this way:<br>
```
python proposed_model.py -test_file <file_name> -false_train_file <file_name> -true_train_file <file_name>  -prediction_file <output_prediction_file_name>
for example:
python proposed_model.py -test_file data/test.fa -false_train_file data/DECOY.tr.fa -true_train_file data/AMP.tr.fa -prediction_file proposed_prediction_output.txt
```
> The true samples in the `data/AMP.tr.fa` file and false samples in the `data/DECOY.tr.fa` will be operated for training. The samples in the `data/test.fa` will be predicted by the trained model. You can get the prediction results in the `prediction_file proposed_prediction_output.txt` of `output` folder.<br>

### proposed fusion model<br>
* The proposed_fusion_model.py can be operated in this way:<br>
```
python proposed_fusion_model.py -test_file <file_name> -false_train_file <file_name> -true_train_file <file_name>  -prediction_file <output_prediction_file_name>
for example:
python proposed_fusion_model.py -test_file data/test.fa -false_train_file data/DECOY.tr.fa -true_train_file data/AMP.tr.fa -prediction_file proposed_prediction_output.txt
```
> The true samples in the `data/AMP.tr.fa` file and false samples in the `data/DECOY.tr.fa` will be operated for training. The samples in the `data/test.fa` will be predicted by the trained model. You can get the prediction results in the `prediction_file proposed_prediction_output.txt` of `output` folder.<br>

* Besides, there are some other options for proposed model and proposed fusion model:<br>
  * `-epochs`: epochs for training<br>
  * `-filter_number`: filter number<br>
  * `-N`: number of different convolutional layers<br>
  * `-embed_length`: the embedding vector dimension of an amino acid<br>
  * `-fix_length`: embedding vector length of a sequence(no less than the maximum length of all sequences)<br>

* For more information, you can type:<br>
```
python proposed_model.py -help
python proposed_fusion_model.py -help
```  
