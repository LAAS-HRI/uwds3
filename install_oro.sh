#!/bin/bash
#
cd $HOME
git clone https://git.openrobots.org/robots/robotpkg.git
echo "export ROBOTPKG_BASE=\"\${HOME}/openrobots\"" >> .bashrc
echo "export PATH=\"\${ROBOTPKG_BASE}/bin:\${ROBOTPKG_BASE}/sbin:\${PATH}\"" >> .bashrc
echo "export PYTHONPATH=\"\${ROBOTPKG_BASE}/lib/python2.7/site-packages:\${PYTHONPATH}\"" >> .bashrc
echo "export ROS_PACKAGE_PATH=\"\${ROBOTPKG_BASE}/src/ros-nodes:\${ROS_PACKAGE_PATH}\"" >> .bashrc
echo "alias oro-server=\"cd ${ROBOTPKG_BASE} && oro-server\"" >> .bashrc
source .bashrc
./bootstrap --prefix=${ROBOTPKG_BASE}

cd $HOME/robotpkg/knowledge/oro-server
echo "Start downloading oro-server..."
make update
cd -
cd $HOME/robotpkg/knowledge/py-oro
echo "Start downloading py-oro..."
make update
cd -
cd $HOME
mkdir -p openrobots/share/ontologies
cd -
echo "Set up the commonsense ontology..."
cp ./models/ontology/commonsense.oro.owl ${ROBOTPKG_BASE}/share/ontologies/
cd ${HOME}
echo "Create an alias for oro-server.."
echo "alias oro-server=\"cd ${ROBOTPKG_BASE} && oro-server\"" >> .bashrc
source .bashrc
echo "Bye bye !"
cd -
