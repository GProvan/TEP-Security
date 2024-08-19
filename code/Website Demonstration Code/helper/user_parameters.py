import streamlit as st
def training_parameters():
	st.sidebar.header('Model Training 🧪')
	st.sidebar.subheader('Use Pretrained Model')
	pretrained = st.sidebar.checkbox('Use Pretrained Model', value=True)
	st.sidebar.subheader('Set Model Parameters')
	batch_size = st.sidebar.slider('Select the batch size', 32, 512, 256, 32)
	epochs = st.sidebar.slider('Select the number of epochs', 5, 25, 10, 1)
	learning_rate = st.sidebar.selectbox('Select the learning rate', [0.001, 0.00001, 0.0001, 0.01])
	hidden_size = st.sidebar.slider('Select the hidden size', 10, 35, 25, 5)
	num_layers = st.sidebar.slider('Select the number of layers', 1, 3, 1, 1)
	sequence_length = st.sidebar.slider('Select the sequence length', 10, 50, 20, 5)
	dropout = st.sidebar.slider('Select the dropout', 0.1, 0.5, 0.2, 0.1)
	use_bias = st.sidebar.checkbox('Use Bias', value=True)
	return pretrained, batch_size, epochs, learning_rate, hidden_size, num_layers, sequence_length, dropout, use_bias

# pretrained, batch_size, epochs, learning_rate, hidden_size, num_layers, sequence_length, dropout, use_bias = training_parameters()