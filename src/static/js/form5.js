$(document).ready(function() {

	$('form').on('submit', function(event) {

		console.log('version 6');

		$.ajax({
			data: $('form').serialize(),

			type : 'POST',
			url : '/process'
		})
		.done(function(data) {
			console.log('here is .done')

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
