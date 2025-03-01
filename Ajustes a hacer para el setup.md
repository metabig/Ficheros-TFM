Se proveen los cambios en el repo para lograr el funcionamiento correcto del entrenamiento.

- requirements.txt: incongruencias entre versiones
- specs_saus.json: parámetros del entrenamiento (con memoria ram pequeña se tiene que ajustar el tamaño del vocabulario en el parámetro vocab_size). Si sale Killed en la ejecución significa que se tiene que reducir el vocab_size o aumentar la memoria RAM.
- train_cl.py, test_cl.py: contienen cambios en las importaciones.

También se puede ver en el output.json la respuesta del modelo a input.xls.