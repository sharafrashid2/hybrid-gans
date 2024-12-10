import os
import numpy as np
import tensorflow.compat.v1 as tf
import random
from rollout import ROLLOUT
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from value_network import ValueNetwork  # For StepGAN
import pickle

# Toggle StepGAN or SeqGAN
use_stepgan = True  # Set to True for StepGAN, False for SeqGAN
# Toggle RelGAN or SeqGAN Generator
use_relgan = True  # Set to True to use RelGAN generator logic

# Hyper-parameters
if use_relgan:
  EMB_DIM = 2048
  HIDDEN_DIM = 2048
else:
  EMB_DIM = 32
  HIDDEN_DIM = 32

SEQ_LENGTH = 20
START_TOKEN = 0
PRE_EPOCH_NUM = 140
SEED = 88
BATCH_SIZE = 64

# Discriminator Hyper-parameters
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

# Training Parameters
TOTAL_BATCH = 200
positive_file = '/content/drive/MyDrive/Project_1_Aneesh_Phishing/Sharaf_Rashid/stepgan/save/data/mosscap_tokens_number.txt'
negative_file = '/content/drive/MyDrive/Project_1_Aneesh_Phishing/Sharaf_Rashid/stepgan/save/generator_sample.txt'
eval_file = '/content/drive/MyDrive/Project_1_Aneesh_Phishing/Sharaf_Rashid/stepgan/save/eval_file.txt'
generated_num = 10000

# Paths to save model weights and generated sequences
save_dir = "/content/drive/MyDrive/Project_1_Aneesh_Phishing/Sharaf_Rashid/stepgan/save/model_weights_relstepgan/"
generated_sequences_dir = "/content/drive/MyDrive/Project_1_Aneesh_Phishing/Sharaf_Rashid/stepgan/save/generated_sequences_relstepgan/"



def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for sequence in generated_samples:
            buffer = ' '.join([str(x) for x in sequence]) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for _ in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def save_model_weights(sess, saver, checkpoint_path):
    print(f"Saving model weights to {checkpoint_path}...")
    saver.save(sess, checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")


def main():
    print("Began")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    assert START_TOKEN == 0

    # Data loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    vocab_size = 1661

    # Initialize models
    generator = Generator(
        vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,
        use_relgan=use_relgan
    )
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                   embedding_size=dis_embedding_dim, filter_sizes=dis_filter_sizes,
                                   num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # SeqGAN rollout or StepGAN value network
    if use_stepgan:
        value_network = ValueNetwork(sequence_length=SEQ_LENGTH, hidden_dim=HIDDEN_DIM, vocab_size=vocab_size)
    else:
        rollout = ROLLOUT(generator, 0.8)

    # TensorFlow session setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Preprocess training data
    print("Creating generator data batches...")
    gen_data_loader.create_batches(positive_file)

    # Saver for saving weights
    saver_generator = tf.train.Saver(var_list=generator.g_params)
    saver_discriminator = tf.train.Saver(var_list=discriminator.params)

    # Pre-train generator
    print('Start pre-training generator...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            print(f'Pre-train epoch {epoch}, generator loss: {loss}')

    # Pre-train discriminator
    print('Start pre-training discriminator...')
    for _ in range(50):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                sess.run(discriminator.train_op, feed)

    # Adversarial Training
    for total_batch in range(TOTAL_BATCH):
        if use_stepgan:
            # StepGAN Training
            for _ in range(3):  # Train Value Network
                samples = gen_data_loader.next_batch()
                q_values = value_network.compute_q_values(sess, samples, discriminator)
                v_loss = value_network.train(sess, samples, q_values)

            # Generator Training with StepGAN Rewards
            samples = generator.generate(sess)
            q_values = value_network.compute_q_values(sess, samples, discriminator)
            v_values = value_network.predict(sess, samples)
            stepgan_rewards = q_values - v_values
            feed = {generator.x: samples, generator.rewards: stepgan_rewards}
            sess.run(generator.g_updates, feed_dict=feed)
        else:
            # SeqGAN Training
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            sess.run(generator.g_updates, feed_dict=feed)

        # Train Discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    sess.run(discriminator.train_op, feed)

        # Save generated sequences and model weights every 1 epoch
        generated_file = generated_sequences_dir + f"generated_epoch_{total_batch}.txt"
        generate_samples(sess, generator, BATCH_SIZE, generated_num, generated_file)
        save_model_weights(sess, saver_generator, save_dir + "generator_epoch_{total_batch}.ckpt")
        save_model_weights(sess, saver_discriminator, save_dir + f"discriminator_epoch_{total_batch}.ckpt")

        # Log progress
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            print(f'Total Batch {total_batch}: Training in progress')

    print('Adversarial training completed.')


if __name__ == '__main__':
    main()