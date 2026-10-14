#pragma once
#include <cmath>
#include <cstdlib>
#include <vector>

inline float Tanh(float x) { return std::tanh(x); }
inline float TanhDerivative(float x) { float t = std::tanh(x); return 1.0f - t * t; }


struct RegressionWeights {
    float* W_output;  //çýkýþ katmaný aðýrlýklarý
    float* b_output;  // bias deðerleri
};

struct RegressionConfig {
    int inputDim;
    int outputDim;
};

struct NetworkWeights { 
    float* W_output;
    float* b_output;
};

struct NetworkConfig { 
    int inputDim;
    int outputDim;
};


struct DynamicRegressionWeights {
    int numLayers;
    std::vector<int> layerSizes;
    std::vector<float*> W;
    std::vector<float*> b;

    DynamicRegressionWeights(int inputDim, std::vector<int> hiddenLayers, int outputDim = 1) {
        numLayers = static_cast<int>(hiddenLayers.size()) + 1;
        layerSizes.push_back(inputDim);
        for (int h : hiddenLayers) layerSizes.push_back(h);
        layerSizes.push_back(outputDim);

        for (int l = 0; l < numLayers; l++) {
            int inputSize = layerSizes[l];
            int outputSize = layerSizes[l + 1];
            float* weights = new float[outputSize * inputSize];
            float* biases = new float[outputSize];
            //giriþ ve çýkýþ nöron sayýlarýna göre ölçek belirleme
            float scale = std::sqrt(2.0f / (inputSize + outputSize));
            for (int i = 0; i < outputSize * inputSize; i++)
                weights[i] = scale * (((float)rand() / RAND_MAX) - 0.5f);
            for (int i = 0; i < outputSize; i++)
                biases[i] = 0.01f;

            W.push_back(weights);
            b.push_back(biases);
        }
    }
    ~DynamicRegressionWeights() {
        for (float* w : W) delete[] w;
        for (float* bias : b) delete[] bias;
    }
};

struct DynamicMultiLayerWeights

{
    int numLayers;                  //katman sayýsý
    std::vector<int> layerSizes;    //her katmandaki nöron sayýlarý
    std::vector<float*> W;          // her katmanýn aðýrlýðý
    std::vector<float*> b;          // her katmandaki bias 

    DynamicMultiLayerWeights(int inputDim, std::vector<int> hiddenLayers, int outputDim) {
        numLayers = static_cast<int>(hiddenLayers.size()) + 1;
        layerSizes.push_back(inputDim);
        for (int h : hiddenLayers) layerSizes.push_back(h);
        layerSizes.push_back(outputDim);

        for (int l = 0; l < numLayers; l++) {
            int inputSize = layerSizes[l];               //önceki katmandaki nöron sayýsý
            int outputSize = layerSizes[l + 1];          //sonraki katmandaki nöron sayýsý 
            float* weights = new float[outputSize * inputSize];   //bellek ayrýmý
            float* biases = new float[outputSize];                //bellek ayrýmý

            for (int i = 0; i < outputSize * inputSize; i++)
                weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
            for (int i = 0; i < outputSize; i++)
                biases[i] = ((float)rand() / RAND_MAX) - 0.5f;

            W.push_back(weights);
            b.push_back(biases);
        }
    }
    ~DynamicMultiLayerWeights() {
        for (float* w : W) delete[] w;
        for (float* bias : b) delete[] bias;
    }
};

struct DynamicMultiLayerConfig {
    int inputDim;
    std::vector<int> hiddenLayers;
    int outputDim;
};



struct RegressionMomentumBuffers {
    std::vector<float*> velocity_W;
    std::vector<float*> velocity_b;
    float momentum_coefficient;

    RegressionMomentumBuffers(DynamicRegressionWeights* weights, float momentum = 0.9f) {
        momentum_coefficient = momentum;

        for (int l = 0; l < weights->numLayers; l++) {
            int inputSize = weights->layerSizes[l];
            int outputSize = weights->layerSizes[l + 1];
            float* vel_w = new float[outputSize * inputSize];
            float* vel_b = new float[outputSize];
            for (int i = 0; i < outputSize * inputSize; i++) vel_w[i] = 0.0f;
            for (int i = 0; i < outputSize; i++) vel_b[i] = 0.0f;
            velocity_W.push_back(vel_w);
            velocity_b.push_back(vel_b);
        }
    }
    ~RegressionMomentumBuffers() {
        for (float* v : velocity_W) delete[] v;
        for (float* v : velocity_b) delete[] v;
    }
};

struct DynamicMomentumBuffers {
    std::vector<float*> velocity_W;  //aðýrlýk hýzlarý
    std::vector<float*> velocity_b;  // bias hýzlarý
    float momentum_coefficient;      //momentum katsayýsý (varsayýlan 0.9 seçili)

    DynamicMomentumBuffers(DynamicMultiLayerWeights* weights, float momentum = 0.9f) {
        momentum_coefficient = momentum;
        for (int l = 0; l < weights->numLayers; l++) {
            int inputSize = weights->layerSizes[l];       //önceki katmandaki nöron sayýsý
            int outputSize = weights->layerSizes[l + 1];  //sonraki katmandaki nöron sayýsý
            float* vel_w = new float[outputSize * inputSize];  //bellek ayýrma
            float* vel_b = new float[outputSize];              //bellek ayýrma
            for (int i = 0; i < outputSize * inputSize; i++) vel_w[i] = 0.0f;
            for (int i = 0; i < outputSize; i++) vel_b[i] = 0.0f;
            velocity_W.push_back(vel_w);
            velocity_b.push_back(vel_b);
        }
    }
    ~DynamicMomentumBuffers() {
        for (float* v : velocity_W) delete[] v;
        for (float* v : velocity_b) delete[] v;
    }
};


inline int TrainingSingleLayerRegression(
    float* Samples, float* targets, int numSample,
    RegressionConfig config, RegressionWeights* weights,
    float learningRate, int maxEpoch,
    float* mean_x, float* std_x,
    float mean_y, float std_y,
    float* errorHistory,
    float momentum, int patience, float improvementThreshold)
{
    int inputDim = config.inputDim;
    int outputDim = config.outputDim;

    //momentum hýzlarý
    float* velocity_W = nullptr;
    float* velocity_b = nullptr;
    //momentum kullanýlacaksa, önceki hýzlarý saklamak için açýlan bellek
    if (momentum > 0.0f) {
        velocity_W = new float[outputDim * inputDim]();  //aðýrlýk hýzlarý
        velocity_b = new float[outputDim]();             //bias hýzlarý
    }

    int epoch = 0;
    float minError = 1e6f;   //erken durdurma için takip edilen en küçük hata
    int patienceCounter = 0; //hata tekrarý sayacý

    while (epoch < maxEpoch) {
        float total_mse = 0.0f;

        for (int step = 0; step < numSample; step++) {
            //forward pass
            float net = 0.0f;
            for (int j = 0; j < inputDim; j++) {
                //giriþ normalize edilir
                float xj = (Samples[step * inputDim + j] - mean_x[j]) / std_x[j];
                // Net = seri( (Giriþ * Aðýrlýk) )
                net += weights->W_output[j] * xj;
            }
            //bias eklenir  Net = (W * X) + b
            net += weights->b_output[0];

            //regresyonda aktivasyon yok(Linear)
            float output = net;
            //hedef deðer normalizasyonu
            float y_normalized = (targets[step] - mean_y) / std_y;
            //hata = hedef - tahmin
            float error = y_normalized - output;
            total_mse += error * error;

            //linear çýkýþ olduðu için türev 1'dir. Delta doðrudan hataya eþit çýkar
            float delta = error;

            //aðýrlýklarý güncelleme
            for (int j = 0; j < inputDim; j++) {
                float xj = (Samples[step * inputDim + j] - mean_x[j]) / std_x[j];
                // Gradyan = Delta * Giriþ
                float gradient = delta * xj;

                if (momentum > 0.0f) {
                    velocity_W[j] = momentum * velocity_W[j] + learningRate * gradient;
                    weights->W_output[j] += velocity_W[j];
                }
                else {
                    weights->W_output[j] += learningRate * gradient;
                }
            }

            if (momentum > 0.0f) {
                velocity_b[0] = momentum * velocity_b[0] + learningRate * delta;
                weights->b_output[0] += velocity_b[0];
            }
            else {
                weights->b_output[0] += learningRate * delta;
            }
        }

        //epoch sonu ortalama hata kaydedilir
        if (errorHistory) errorHistory[epoch] = total_mse / numSample;

        if (errorHistory && errorHistory[epoch] < minError - improvementThreshold) {
            minError = errorHistory[epoch];
            patienceCounter = 0;
        }
        else {
            patienceCounter++;
        }

        if (patienceCounter >= patience) break;
        epoch++;
    }

    //hýz deðerlerini temizleme
    if (velocity_W) delete[] velocity_W;
    if (velocity_b) delete[] velocity_b;
    return (epoch < maxEpoch) ? epoch + 1 : maxEpoch;
}


inline int TrainingMultiLayerRegression(
    float* Samples, float* targets, int numSample,
    DynamicRegressionWeights* weights,
    float learningRate, int maxEpoch,
    float* mean_x, float* std_x,
    float mean_y, float std_y,
    float* errorHistory,
    float momentum, int patience, float improvementThreshold)
{
    int inputDim = weights->layerSizes[0];
    int numLayers = weights->numLayers;

    std::vector<float*> activations;
    std::vector<float*> derivatives;
    std::vector<float*> deltas;

    activations.push_back(new float[inputDim]);
    for (int l = 0; l < numLayers; l++) {
        int size = weights->layerSizes[l + 1];
        activations.push_back(new float[size]);
        derivatives.push_back(new float[size]);
        deltas.push_back(new float[size]);
    }

    RegressionMomentumBuffers* momentumBuf = nullptr;
    if (momentum > 0.0f) momentumBuf = new RegressionMomentumBuffers(weights, momentum);

    int epoch = 0;
    float minError = 1e6f;
    int patienceCounter = 0;

    while (epoch < maxEpoch) {
        float total_mse = 0.0f;

        for (int step = 0; step < numSample; step++) {
            
            for (int i = 0; i < inputDim; i++)
                activations[0][i] = (Samples[step * inputDim + i] - mean_x[i]) / std_x[i];

            for (int l = 0; l < numLayers - 1; l++) {
                int inputSize = weights->layerSizes[l];
                int outputSize = weights->layerSizes[l + 1];
                for (int j = 0; j < outputSize; j++) {
                    float net = 0.0f;
                    for (int i = 0; i < inputSize; i++)
                        net += weights->W[l][j * inputSize + i] * activations[l][i];
                    net += weights->b[l][j];
                    activations[l + 1][j] = Tanh(net);
                    derivatives[l][j] = TanhDerivative(net);
                }
            }

            int lastLayer = numLayers - 1;
            int inputSize = weights->layerSizes[lastLayer];
            float net = 0.0f;
            for (int i = 0; i < inputSize; i++)
                net += weights->W[lastLayer][i] * activations[lastLayer][i];
            net += weights->b[lastLayer][0];
            activations[numLayers][0] = net;
            derivatives[lastLayer][0] = 1.0f;

            float y_normalized = (targets[step] - mean_y) / std_y;
            float error = y_normalized - activations[numLayers][0];
            deltas[lastLayer][0] = error;
            total_mse += error * error;

            for (int l = numLayers - 2; l >= 0; l--) {
                int currentSize = weights->layerSizes[l + 1];
                int nextSize = weights->layerSizes[l + 2];
                for (int j = 0; j < currentSize; j++) {
                    float error_sum = 0.0f;
                    for (int k = 0; k < nextSize; k++)
                        error_sum += deltas[l + 1][k] * weights->W[l + 1][k * currentSize + j];
                    deltas[l][j] = error_sum * derivatives[l][j];
                }
            }

            for (int l = 0; l < numLayers; l++) {
                int inpSize = weights->layerSizes[l];
                int outSize = weights->layerSizes[l + 1];
                for (int j = 0; j < outSize; j++) {
                    for (int i = 0; i < inpSize; i++) {
                        int idx = j * inpSize + i;
                        float grad = deltas[l][j] * activations[l][i];
                        if (momentumBuf) {
                            momentumBuf->velocity_W[l][idx] = momentum * momentumBuf->velocity_W[l][idx] + learningRate * grad;
                            weights->W[l][idx] += momentumBuf->velocity_W[l][idx];
                        }
                        else {
                            weights->W[l][idx] += learningRate * grad;
                        }
                    }
                    if (momentumBuf) {
                        momentumBuf->velocity_b[l][j] = momentum * momentumBuf->velocity_b[l][j] + learningRate * deltas[l][j];
                        weights->b[l][j] += momentumBuf->velocity_b[l][j];
                    }
                    else {
                        weights->b[l][j] += learningRate * deltas[l][j];
                    }
                }
            }
        }

        if (errorHistory) errorHistory[epoch] = total_mse / numSample;

        if (errorHistory && errorHistory[epoch] < minError - improvementThreshold) {
            minError = errorHistory[epoch];
            patienceCounter = 0;
        }
        else {
            patienceCounter++;
        }

        if (patienceCounter >= patience) break;
        epoch++;
    }

    for (float* a : activations) delete[] a;
    for (float* d : derivatives) delete[] d;
    for (float* d : deltas) delete[] d;
    if (momentumBuf) delete momentumBuf;
    return (epoch < maxEpoch) ? epoch + 1 : maxEpoch;
}


inline int TrainingSingleLayer(
    float* Samples, float* targets, int numSample,
    NetworkConfig config, NetworkWeights* weights,
    float learningRate, int maxEpoch,
    float* mean, float* std,
    float* errorHistory,
    float momentum, int patience, float improvementThreshold)
{
    int inputDim = config.inputDim;
    int numOutNeuron = config.outputDim;

    float* net = new float[numOutNeuron];  //net giriþ toplamý
    float* out = new float[numOutNeuron];  //aktivasyon sonrasý çýkýþ
    float* out_der = new float[numOutNeuron];  //türev deðerleri
    float* delta = new float[numOutNeuron];    //hata düzeltme deðerleri

    float* velocity_W = nullptr;
    float* velocity_b = nullptr;
    if (momentum > 0.0f) {
        velocity_W = new float[numOutNeuron * inputDim];
        velocity_b = new float[numOutNeuron];
        for (int i = 0; i < numOutNeuron * inputDim; i++) velocity_W[i] = 0.0f;
        for (int i = 0; i < numOutNeuron; i++) velocity_b[i] = 0.0f;
    }

    int epoch = 0;
    float minError = 1e6f;
    int patienceCounter = 0;

    while (epoch < maxEpoch) {
        float total_err = 0.0f;

        for (int step = 0; step < numSample; step++) {
            //forward pass
            // her k çýkýþ nöronu için hesapla
            for (int k = 0; k < numOutNeuron; k++) {
                net[k] = 0.0f;
                for (int j = 0; j < inputDim; j++) {
                    //giriþi normalize et
                    float xj = (Samples[step * inputDim + j] - mean[j]) / std[j];
                    // Aðýrlýk * Giriþ
                    // Tek nöron varsa dizi indeksi [j], çok nöron varsa [k*inputDim + j]
                    if (numOutNeuron == 1)
                        net[k] += weights->W_output[j] * xj;
                    else
                        net[k] += weights->W_output[k * inputDim + j] * xj;
                }
                //bias ekleme
                net[k] += (numOutNeuron == 1) ? weights->b_output[0] : weights->b_output[k];
                //Tanh aktivasyonu
                out[k] = Tanh(net[k]);
                //TÜREV: Tanh türevini hesapla(1 - out ^ 2).
                out_der[k] = TanhDerivative(net[k]);
            }
            //backpropagation
            for (int k = 0; k < numOutNeuron; k++) {
                // Eðer nöron indexi (k), hedef etikete (targets[step]) eþitse hedef +1, deðilse -1
                float d_k = (numOutNeuron == 1)
                    ? ((targets[step] == 1.0f) ? 1.0f : -1.0f)
                    : ((targets[step] == k) ? 1.0f : -1.0f);

                float e_k = d_k - out[k];
                // Delta = Hata * Türev
                delta[k] = e_k * out_der[k];

                for (int j = 0; j < inputDim; j++) {
                    float xj = (Samples[step * inputDim + j] - mean[j]) / std[j];
                    float gradient = delta[k] * xj;
                    int idx = (numOutNeuron == 1) ? j : (k * inputDim + j);

                    if (momentum > 0.0f) {
                        velocity_W[idx] = momentum * velocity_W[idx] + learningRate * gradient;
                        weights->W_output[idx] += velocity_W[idx];
                    }
                    else {
                        weights->W_output[idx] += learningRate * gradient;
                    }
                }

                int bias_idx = (numOutNeuron == 1) ? 0 : k;
                if (momentum > 0.0f) {
                    velocity_b[bias_idx] = momentum * velocity_b[bias_idx] + learningRate * delta[k];
                    weights->b_output[bias_idx] += velocity_b[bias_idx];
                }
                else {
                    weights->b_output[bias_idx] += learningRate * delta[k];
                }

                total_err += (e_k * e_k);
            }
        }

        if (errorHistory) errorHistory[epoch] = total_err / numSample;

        if (errorHistory && errorHistory[epoch] < minError - improvementThreshold) {
            minError = errorHistory[epoch];
            patienceCounter = 0;
        }
        else {
            patienceCounter++;
        }

        if (patienceCounter >= patience) break;
        epoch++;
    }

    delete[] net; delete[] out; delete[] out_der; delete[] delta;
    if (velocity_W) delete[] velocity_W;
    if (velocity_b) delete[] velocity_b;
    return (epoch < maxEpoch) ? epoch + 1 : maxEpoch;
}


inline int TrainingMultiLayer(
    float* Samples, float* targets, int numSample,
    DynamicMultiLayerConfig config, DynamicMultiLayerWeights* weights,
    float learningRate, int maxEpoch,
    float* mean, float* std,
    float* errorHistory,
    float momentum, int patience, float improvementThreshold)
{
    int inputDim = config.inputDim;
    int outputDim = config.outputDim;
    int numLayers = weights->numLayers;

    //dinamik vektörler
    std::vector<float*> activations;  //nöronlarýn çýkýþ deðerleri
    std::vector<float*> derivatives;  // türev deðerleri (1 - tanh**2)
    std::vector<float*> deltas;       // her nöronun hataya katkýsý

    //giriþ katmaný için ayrýlan yer
    activations.push_back(new float[inputDim]);

    //gizli katmanlar ve çýkýþ katmaný için ayrýlan yerler
    for (int l = 0; l < numLayers; l++) {
        int size = weights->layerSizes[l + 1];    // katmandaki nöron sayýsý
        activations.push_back(new float[size]);
        derivatives.push_back(new float[size]);
        deltas.push_back(new float[size]);
    }

    //momentum aktifse önceki hýzlarý tutacak buffer
    DynamicMomentumBuffers* momentumBuf = nullptr;
    if (momentum > 0.0f) momentumBuf = new DynamicMomentumBuffers(weights, momentum);

    int epoch = 0;
    float minError = 1e6f;
    int patienceCounter = 0;

    while (epoch < maxEpoch) {
        float total_err = 0.0f; //mevcut epochtaki hatayý sýfýrladýk

        //her bir örnek(step) için yapýlan iþlemler  
        for (int step = 0; step < numSample; step++) {
            //gelen veriyi normalize edip kullanýyoruz
            for (int i = 0; i < inputDim; i++)
                // (deðer - ortalama) / standart sapma
                activations[0][i] = (Samples[step * inputDim + i] - mean[i]) / std[i];

            //forward
            for (int l = 0; l < numLayers; l++) {  //katman
                int inpSize = weights->layerSizes[l];     //þu snki katman boyutu
                int outSize = weights->layerSizes[l + 1]; //sonraki katman boyutu
                // þu anki katmandaki her bir nöron için hesaplama
                for (int j = 0; j < outSize; j++) { // satýr    
                    float net = 0.0f;
                    //bir önceki katmandaki tüm nöronlarýn çýkýþlarý aðýrlýklarla çarpýp toplanýr
                    // Net = toplam(Input * Weight) + Bias 
                    for (int i = 0; i < inpSize; i++) // sütun     
                        net += weights->W[l][j * inpSize + i] * activations[l][i];
                    net += weights->b[l][j]; // +bias

                    activations[l + 1][j] = Tanh(net); //tanh fonksiyonundan geçiriliyor
                    derivatives[l][j] = TanhDerivative(net); //geri yayýlým(backpropagation) için türev hesabý
                }
            }

            //hata
            int lastLayer = numLayers - 1;
            for (int j = 0; j < outputDim; j++) {  //hedef nöron
                //hedef deðerler
                //eðer sýnýflandýrma ise hedef 1 diðerleri -1
                float desired = (outputDim == 1)
                    ? ((targets[step] == 1.0f) ? 1.0f : -1.0f)
                    : ((targets[step] == j) ? 1.0f : -1.0f);
                //hata hesabý = hedef - tahmin
                float error = desired - activations[numLayers][j];
                // delta = hata * aktivasyonun türevi
                deltas[lastLayer][j] = error * derivatives[lastLayer][j];
                total_err += (error * error);
            }
            
            //backward
            for (int l = numLayers - 2; l >= 0; l--) {  //geriye doðru katman döngüsü
                int currSize = weights->layerSizes[l + 1]; //þu an hesaplanan katman
                int nxtSize = weights->layerSizes[l + 2];  //hatanýn geldiði sonraki katman
                for (int j = 0; j < currSize; j++) {
                    float error_sum = 0.0f;
                    //sonraki katmandaki deltalar, aradaki aðýrlýklar aðýrlandýrýlarak toplanýr
                    for (int k = 0; k < nxtSize; k++)
                        error_sum += deltas[l + 1][k] * weights->W[l + 1][k * currSize + j];
                    // mevcut katmanýn deltasý = gelen hata toplamý * kendi türevi
                    deltas[l][j] = error_sum * derivatives[l][j];
                }
            }

            // tüm katmanlarý gezerek aðýrlýklarý deðiþtiriyoruz
            for (int l = 0; l < numLayers; l++) {
                int inpSize = weights->layerSizes[l];
                int outSize = weights->layerSizes[l + 1];
                for (int j = 0; j < outSize; j++) {       //hedef nöron
                    for (int i = 0; i < inpSize; i++) {   //kaynak nöron
                        int idx = j * inpSize + i;        //aðýrlýk dizisindeki indeks
                        // gradyan = delta * giriþ deðeri(hatanýn yönü)
                        float grad = deltas[l][j] * activations[l][i];
                        if (momentumBuf) {
                            //yeni hýz = (momentum * eski hýz) + (learningRate * gradyan)
                            momentumBuf->velocity_W[l][idx] = momentum * momentumBuf->velocity_W[l][idx] + learningRate * grad;
                            // aðýrlýk hýza göre güncellenir
                            weights->W[l][idx] += momentumBuf->velocity_W[l][idx];
                        }
                        else {
                            weights->W[l][idx] += learningRate * grad;
                        }
                    }
                    //bias güncellemesi
                    if (momentumBuf) {
                        momentumBuf->velocity_b[l][j] = momentum * momentumBuf->velocity_b[l][j] + learningRate * deltas[l][j];
                        weights->b[l][j] += momentumBuf->velocity_b[l][j];
                    }
                    else {
                        weights->b[l][j] += learningRate * deltas[l][j];
                    }
                }
            }
        }

        if (errorHistory) errorHistory[epoch] = total_err / numSample;

        if (errorHistory && errorHistory[epoch] < minError - improvementThreshold) {
            minError = errorHistory[epoch];
            patienceCounter = 0;
        }
        else {
            patienceCounter++;
        }
        if (patienceCounter >= patience) break;
        epoch++;
    }

    for (float* a : activations) delete[] a;
    for (float* d : derivatives) delete[] d;
    for (float* d : deltas) delete[] d;
    if (momentumBuf) delete momentumBuf;
    return (epoch < maxEpoch) ? epoch + 1 : maxEpoch;
}



inline float TestSingleLayerRegression(
    float* x, RegressionWeights* weights, RegressionConfig config,
    float* mean_x, float* std_x, float mean_y, float std_y)
{
    float net = 0.0f; //net deðer

    for (int j = 0; j < config.inputDim; j++) {
        //normalizasyon
        float xj = (x[j] - mean_x[j]) / std_x[j];
        // Net += Aðýrlýk * NormalizeGiriþ
        net += weights->W_output[j] * xj;
    }
    // bias ekleme
    net += weights->b_output[0];
    // Gerçek = (NormalizeDeðer * StandartSapma) + Ortalama
    return net * std_y + mean_y;
}

inline float TestMultiLayerRegression(
    float* x, DynamicRegressionWeights* weights,
    float* mean_x, float* std_x, float mean_y, float std_y)
{
    int inputDim = weights->layerSizes[0];
    int numLayers = weights->numLayers;
    //her katmanýn çýkýþýný saklayan vektör
    std::vector<float*> activations;
    //giriþ kataný için ayrýlan yer
    activations.push_back(new float[inputDim]);
    //diðer katmanlar için ayrýlan yer
    for (int l = 0; l < numLayers; l++) activations.push_back(new float[weights->layerSizes[l + 1]]);
    //giriþ normalize edilir ve ilk katmana yüklenir
    for (int i = 0; i < inputDim; i++) activations[0][i] = (x[i] - mean_x[i]) / std_x[i];
    //gizli katmanlar arasý geçiþ
    for (int l = 0; l < numLayers - 1; l++) {
        int inpSize = weights->layerSizes[l];
        int outSize = weights->layerSizes[l + 1];
        for (int j = 0; j < outSize; j++) {
            float net = 0.0f;
            //matris çarpýmý: önceki katman çýkýþý * aðýrlýklar
            for (int i = 0; i < inpSize; i++)
                net += weights->W[l][j * inpSize + i] * activations[l][i];
            net += weights->b[l][j]; //bias ekler
            activations[l + 1][j] = Tanh(net);
        }
    }

    int last = numLayers - 1;
    int inpSize = weights->layerSizes[last];
    float net = 0.0f;
    for (int i = 0; i < inpSize; i++)
        net += weights->W[last][i] * activations[last][i];
    net += weights->b[last][0];

    float result = net * std_y + mean_y;
    for (float* a : activations) delete[] a;
    return result;
}

inline int TestSingleLayer(float* x, NetworkWeights* weights, NetworkConfig config, float* mean, float* std) {
    float* output = new float[config.outputDim];
    //her bir sýnýf nöronu için hesap
    for (int i = 0; i < config.outputDim; i++) {
        output[i] = 0.0f;
        //giriþ * aðýrlýk
        for (int j = 0; j < config.inputDim; j++) {
            float xj = (x[j] - mean[j]) / std[j];  //normalizasyon
            if (config.outputDim == 1)
                output[i] += weights->W_output[j] * xj; 
            else
                output[i] += weights->W_output[i * config.inputDim + j] * xj;
        }
        output[i] += (config.outputDim == 1) ? weights->b_output[0] : weights->b_output[i];
        output[i] = Tanh(output[i]);
    }

    int index_max = 0;
    if (config.outputDim == 1) {
        index_max = (output[0] > 0.0f) ? 0 : 1;
    }
    else {
        //en yüksek skoru üreten nöron
        float max_val = output[0];
        for (int i = 1; i < config.outputDim; i++) {
            if (output[i] > max_val) {
                max_val = output[i]; 
                index_max = i; //kazanan sýnýfýn endeksi
            }
        }
    }
    delete[] output;
    return index_max;
}

inline int TestMultiLayer(
    float* x, DynamicMultiLayerWeights* weights, DynamicMultiLayerConfig config, float* mean, float* std)
{
    int inputDim = config.inputDim;
    int outputDim = config.outputDim;
    int numLayers = weights->numLayers;

    std::vector<float*> activations;
    activations.push_back(new float[inputDim]);
    for (int l = 0; l < numLayers; l++) activations.push_back(new float[weights->layerSizes[l + 1]]);

    for (int i = 0; i < inputDim; i++) activations[0][i] = (x[i] - mean[i]) / std[i];

    for (int l = 0; l < numLayers; l++) {
        int inpSize = weights->layerSizes[l];
        int outSize = weights->layerSizes[l + 1];
        for (int j = 0; j < outSize; j++) {
            float net = 0.0f;
            for (int i = 0; i < inpSize; i++)
                net += weights->W[l][j * inpSize + i] * activations[l][i];
            net += weights->b[l][j];
            activations[l + 1][j] = Tanh(net);
        }
    }

    int maxIndex = 0;
    if (outputDim > 1) {
        float maxVal = activations[numLayers][0];
        for (int i = 1; i < outputDim; i++) {
            if (activations[numLayers][i] > maxVal) {
                maxVal = activations[numLayers][i];
                maxIndex = i;
            }
        }
    }
    else {
        maxIndex = (activations[numLayers][0] > 0.0f) ? 0 : 1;
    }

    for (float* a : activations) delete[] a;
    return maxIndex;
}