rm -rf outlines || echo "rm outlines fail"
rm -rf interegular || echo "rm interegular fail"
rm -rf vllm-ascend || echo "rm interegular fail"
rm -rf hb-serve-chat-npu_R25C20-arm-o1.tar || echo "rm hb-serve-chat-npu_R25C20-arm-o1.tar fail"
docker rmi -f docker.das-security.cn/hb/hb-serve-chat-npu:R25C20-arm-o1 || echo "docker rmi fail"

git clone http://gitlab.info.dbappsecurity.com.cn/da.chen/eight-horses.git

cd eight-horses
git checkout vllm-0.8.5.post1-dev
cd ..
mv eight-horses vllm


docker build -f vllm/Dockerfile.gpu -t docker.das-security.cn/hb/hb-serve-chat-gpu:R26C10-v4.1.0-o1 .
#docker save -o hb-serve-chat-npu_R25C20-arm-o1.tar docker.das-security.cn/hb/hb-serve-chat-npu:R25C20-arm-o4
#docker push docker.das-security.cn/hb/hb-serve-chat-npu:R25C20-arm-o4