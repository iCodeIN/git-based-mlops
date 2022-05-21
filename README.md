# Git(Hub) Based MLOps

This project shows how to achieve MLOps using tools such as [DVC](https://dvc.org/), [DVC Studio](https://studio.iterative.ai/), [DVCLive](https://dvc.org/doc/dvclive), [CML](https://cml.dev/), (possibly)[GTO](https://github.com/iterative/gto) - all products built by [iterative.ai](https://iterative.ai/), [Google Drive](https://www.google.com/drive/), and [Jarvislabs.ai](https://jarvislabs.ai/). Here is the brief description of each tools
- **DVC(Data Version Control)**: 
- **DVCLive**:
- **DVC Studio**:
- **CML(Continuous Machine Learning)**:
- **GTO(Git Tag Ops)**:
- **Google Drive**:
- **Jarvislabs.ai**: 


Google Drive can be accessed with credential JSON file. It is encrypted using GPG like `echo "PASSPHRASE" | gpg --batch --passphrase-fd 0 -c -o .dvc/tmp/gdrive-user-credentials.json.gpg .dvc/tmp/gdrive-user-credentials.json`, and it can be decrypted with `echo "PASSPHRASE" | gpg --batch --passphrase-fd 0 -d -o .dvc/tmp/gdrive-user-credentials.json .dvc/tmp/gdrive-user-credentials.json.gpg`.
