$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			console.log('Hi');

			data : {
				test: 'this is a test',
				name1 : $('#nameInput').val(),
				name2 : $('#nameInput2').val()
			},
			type : 'POST',
			url : '/process'
		})
		.done(function(data) {
			console.log('here is done')

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(data.test).show();
				$('#errorAlert').hide();
			}

		});

		event.preventDefault();

	});

});
