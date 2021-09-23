
/*event for login*/
$("#reset_password_form").submit(function (e) {
    e.preventDefault();
}).validate({
    rules: {       
        password: {
            required: true,
            minlength: 6
        }
    },
    messages: {       
        password: {
            required: "You must enter a password.",
            minlength: "Your password must be at least {0} characters."
        }
    },
    submitHandler: function (form) {
        $('#loader').show();
        var email = getEmail();    
        var password = $("#password").val();
        var data = { "data": { "email": email, "password": password } };   
        // console.log(data);  
        $.ajax({
            type: "POST",
            url: "/start_reset_password",
            data: JSON.stringify(data),
            contentType: 'application/json;charset=UTF-8',
            dataType: "json",
            success: function (response) {
                $('#loader').hide();
                result = response["data"];
                display_notice(result["success"], result["msg"]);
                window.location.href = result["login_site_url"];
            },
            error: function (err) {
                $('#loader').hide();
                alert_error_ajax();
            }
        });
    },
});

var getEmail = function() {
    var page_url = window.location.href;
    // console.log(page_url);
    var url_parts = page_url.split("/");
    var email = url_parts[url_parts.length - 1];
    return decodeURIComponent(email);
};

///START
var key_enter_evt = function (event) {
    if (event.keyCode === 13) {
        event.preventDefault();
        document.getElementById("btn_login").click();
    }
}

var input = document.getElementById("password");
input.addEventListener("keyup", key_enter_evt);
///END