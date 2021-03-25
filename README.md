# keras-tuner-grpc

```
kompose convert
kubectl apply -f keras-tuner-tunerx-deployment.yaml 
kubectl apply -f keras-tuner-chief-deployment.yaml 
kubectl exec -it keras-tuner-tunerx-cff98c8d8-skcn2 -- /bin/bash
kubectl get pods
kubectl exec -it keras-tuner-chief-6b5986cbdc-jshhh -- /bin/bash
```
