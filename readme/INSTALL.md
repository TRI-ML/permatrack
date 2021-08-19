# Installation


1. We provide a Docker file to re-create the environment which was used in our experiments under `$PermaTrack_ROOT/docker/Dockerfile`. You can either configure the environment yourself using the docker file as a guide or build it via:
  ~~~
    cd $PermaTrack_ROOT
    make docker-build
    make docker-start-interactive
  ~~~ 

2. The only step that has to be done manually is compiling of deformabel convolutions module.

  ~~~
    cd $PermaTrack_ROOT/src/lib/model/networks/
    git clone https://github.com/CharlesShang/DCNv2/ 
    cd DCNv2
    ./make.sh
  ~~~
