Setting up output folder
```
groupadd peak_free_atac

usermod -a -G peak_free_atac wsaelens

setfacl -d -m group:peak_free_atac:rwx /data/peak_free_atac
setfacl -m group:peak_free_atac:rwx /data/peak_free_atac
sudo chown wsaelens /data/peak_free_atac 
sudo chgrp peak_free_atac /data/peak_free_atac 
sudo chmod g+s /data/peak_free_atac # very important
```

Setting up the conda environment

```
conda create --prefix /data/peak_free_atac/software/peak_free_atac python=3.9
echo 'alias conda_peak_free_atac="conda activate /data/peak_free_atac/software/peak_free_atac"' >>~/.bash_profile

source ~/.bash_profile

conda_peak_free_atac
```

Per user
```
echo 'alias conda_peak_free_atac="conda activate /data/peak_free_atac/software/peak_free_atac"' >>~/.bash_profile
source ~/.bash_profile

pip install -e ~/projects/peakcheck/package

ln -s /data/peak_free_atac/output
ln -s /data/peak_free_atac/software
```
