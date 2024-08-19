def generation_explanation():
	from cadlae.preprocess import DataProcessor
	from cadlae.explainer import ActionExplainer
	import streamlit.components.v1 as components
	import streamlit as st
	import seaborn as sns
	sns.set_style("whitegrid")
	
	from helper.st_utils import data_preprocess
	# Data Preprocessing
	train = "data/train_data.csv"
	test = "data/test_data_idv4.csv"
	X_train, y_train, X_test, y_test, col_names, scaler = data_preprocess(train, test)
	from helper.long_form_text import explainer_text
	
	explainer_text()
	
	# streamlit text input type int between 0 and 10
	
	index = st.sidebar.number_input("Enter index of data point to explain", min_value=0, max_value=len(X_train), value=550, step=1)
	
	if st.button("Explain"):
		st.warning('Due to compatibility issues, the following charts are not dynamically sized, as a result, they may not be displayed correctly. We apologise for any inconvenience caused.')
		with st.spinner('Explainer is learning, Please Wait...'):
			mod = ActionExplainer()
			mod.fit(X_test, y_test, max_depth=4)
			mod.learn_data()
		
	
	
		def st_dtree(plot, height=None):
			#dtree_html = f"<body>{plot.view().svg()}</body>"
			dtree_html = f"<div style='text-align:center'><body>{plot.view().svg()}</body></div>"
			
			components.html(dtree_html, height=height, width=750)
		st.subheader('Global Explanation')
		
		st_dtree(mod.model, 550)
		
		def st_dtree_local(plot, height=None,width = None):
			dtree_html = f"<div style='text-align:center'><body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body></div>"
			
			#dtree_html = f"<body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body>"
			
			components.html(dtree_html, height=height, width=width)
		
		
		
		st.subheader('Local Explanation for data point {}'.format(index))
		st_dtree_local(mod.model, 600,800)
		prediction_proba = mod.clf.predict_proba(X_test.iloc[index].values.reshape(1, -1))
		if prediction_proba[0][0] > prediction_proba[0][1]:
			mode = "Normal Activity"
			prob = round(prediction_proba[0][0] * 100, 2)
		else:
			mode = "Fault"
			prob = round(prediction_proba[0][1] * 100, 2)
		
		st.markdown("""
		<style>
		.big-font {
		    font-size:18px !important;
		}
		</style>
		""", unsafe_allow_html=True)

		st.subheader('Report for data point: {} 🔍'.format(index))
	
		st.markdown('<p class="big-font">Model prediction for data point {}: <strong>{}</strong></p>'.format(index,mode), unsafe_allow_html=True)
		st.markdown('<p class="big-font">Probability of data point {} being in class {}: <strong>{}%.</strong></p>'.format(index, mode,prob),unsafe_allow_html=True)
		st.subheader('Suggested Action 💡')
		actions = mod.action(index)
		for action, description in actions.items():
			st.write('<p class="big-font"><strong>{}: </strong>{}</p>'.format(action,description), unsafe_allow_html=True)
		
