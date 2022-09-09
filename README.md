# Hachathon
OLIVIER Antoine
GERARD Romain
BEKKALI Lina

Notre projet a partiellement consisté à faire du fine tuning sur le modèle fasterrcnn_resnet50_fpn de torchvision.
Pour cela nous nous sommes aidés du tutoriel: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html .
Cela nous a permis de créer notre propre modèle "modell10.pth"

Les fichiers python que nous avons créés sont hackhathon_v1.py, hackhathon_v2.py, heatmap_v1.py et precise_heatmap.py. Les autres sont issus du dossier references/detection/ du github de pytorch/vision .

Pour que la manipulation des fichiers Detection_Train_Set et Detection_Test_Set se fassent correctement il faut nommer les fichiers contenant les images "PNGimages" et les fichiers contenant les json "JSONfiles"

Pour la génération de la heatmap, il faut telecharger les dossiers des json pour toutes les caméras, et mettre le nom de la caméra voulue dans le code
