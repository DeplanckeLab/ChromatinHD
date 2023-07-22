version="0.0.18"

git add .
git commit -m "version v${version}"

git tag -a v${version} -m "v${version}"

python -m build

twine upload --repository testpypi dist/chromatinhd-${version}.tar.gz --verbose


twine upload dist/chromatinhd-${version}.tar.gz --verbose