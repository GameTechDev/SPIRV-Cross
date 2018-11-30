///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2017, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spirv_ispc.hpp"
#include "GLSL.std.450.h"
#include "spirv_cfg.hpp"
#include <algorithm>
#include <assert.h>
#include <iomanip> // std::put_time

using namespace spv;
using namespace spirv_cross;
using namespace std;

// pull in from spirv_cross.cpp.
string ensure_valid_identifier(const string &name, bool member);

// Enable this define to print out the full dependancy chains used for determining if variables are vector/varying or scalar/uniform.
// Default assumption is that they are uniform.
#define DUMP_VARYING_DEPENDANCIES 0

void CompilerISPC::emit_buffer_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto instance_name = to_name(var.self);

	emit_block_struct(type);
	auto buffer_name = to_name(type.self);
	statement("");
}

void CompilerISPC::emit_interface_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);

	auto instance_name = to_name(var.self);

	string buffer_name;
	if (meta[type.self].decoration.decoration_flags.get(DecorationBlock))
	{
		emit_block_struct(type);
		buffer_name = to_name(type.self);
	}
	else
		buffer_name = type_to_glsl(type);
}

void CompilerISPC::emit_shared(const SPIRVariable &var)
{
	add_resource_name(var.self);

	statement(CompilerGLSL::variable_decl(var), ";");
}

void CompilerISPC::emit_uniform(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto instance_name = to_name(var.self);

	string type_name = type_to_glsl(type);
	remap_variable_type_name(type, instance_name, type_name);

	statement("");
}

void CompilerISPC::emit_push_constant_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto &flags = meta[var.self].decoration.decoration_flags;
	if ((flags.get(DecorationBinding) || flags.get(DecorationDescriptorSet)))
		SPIRV_CROSS_THROW("Push constant blocks cannot be compiled to GLSL with Binding or Set syntax. "
		                  "Remap to location with reflection API first or disable these decorations.");

	emit_block_struct(type);

	statement("");
}

void CompilerISPC::emit_block_struct(SPIRType &type)
{
	// C++ can't do interface blocks, so we fake it by emitting a separate struct.
	// However, these structs are not allowed to alias anything, so remove it before
	// emitting the struct.
	//
	// The type we have here needs to be resolved to the non-pointer type so we can remove aliases.
	auto &self = get<SPIRType>(type.self);
	self.type_alias = 0;
	emit_struct(self);
}

void CompilerISPC::emit_struct(SPIRType &type)
{
	// Struct types can be stamped out multiple times
	// with just different offsets, matrix layouts, etc ...
	// Type-punning with these types is legal, which complicates things
	// when we are storing struct and array types in an SSBO for example.
	if (type.type_alias != 0)
		return;

	// Don't declare empty structs in GLSL, this is not allowed.
	// Empty structs is a corner case of HLSL output, and only sensible thing to do is avoiding to declare
	// these types.
	if (type_is_empty(type))
		return;

	add_resource_name(type.self);
	auto name = type_to_glsl(type);

	statement(!backend.explicit_struct_type ? "struct " : "", name);
	begin_scope();

	type.member_name_cache.clear();

	uint32_t i = 0;
	bool emitted = false;
	for (auto &member : type.member_types)
	{
		auto &membertype = get<SPIRType>(member);

		if (!ignore_runtimearray_padding)
		{
			// if we are a runtime array that is a struct, we may need padding to conform to GLSL rules
			if (!membertype.array.empty() && !membertype.array.back() && membertype.basetype == SPIRType::Struct)
			{
				// array_size comes from SPIRV, but this may be different to the actual size
				// of the member struct elements that have been declared.
				// ie. a struct may be rounded up to the nearest vec4 size, depending on the padding rules.
				// This is generally only an issue for runtime arrays which require the correct padding so ISPC
				// can iterate through the array correctly.
				auto array_size = get_declared_struct_size(type);
				auto member_size = get_declared_struct_size(membertype);
				auto padding_size = array_size - member_size;
				if (padding_size && meta[membertype.self].decoration.runtime_array_padding != padding_size)
				{
					// flag this type along with the original if it is aliased
					meta[membertype.self].decoration.runtime_array_padding = padding_size;
					if (membertype.type_alias)
						meta[membertype.type_alias].decoration.runtime_array_padding = padding_size;

					// The membertype may have already been written out, so we need to recompile it with the correct padding.
					// The padding meta data will persist through a re-compile.
					force_recompile = true;
				}
			}
		}

		add_member_name(type, i);
		emit_struct_member(type, member, i);
		i++;
		emitted = true;
	}

	if (emitted)
	{
		if (meta[type.self].decoration.runtime_array_padding)
		{
			statement("int8 spirvcross_std430_auto_padding[",
			          to_string(meta[type.self].decoration.runtime_array_padding), "];");
		}
	}
	end_scope_decl();

	if (emitted)
		statement("");
}

void CompilerISPC::emit_specialization_constants()
{
	bool emitted = false;
	for (auto &id : ids)
	{
		if (id.get_type() == TypeConstant)
		{
			auto &c = id.get<SPIRConstant>();
			if (!c.specialization)
				continue;

			auto name = to_name(c.self);

			// Don't support changing specialization constants at runtime, but do support them as #defines.
			statement("#define ", name, " ", constant_expression(c));
			emitted = true;
		}
		else if (id.get_type() == TypeConstantOp)
		{
			auto &c = id.get<SPIRConstantOp>();
			auto &type = get<SPIRType>(c.basetype);
			auto name = to_name(c.self);
			statement("static const uniform ", variable_decl(type, name, 0), " = ", constant_op_expression(c), ";");
			emitted = true;
		}
	}

	if (emitted)
		statement("");
}

string CompilerISPC::bitcast_glsl_op(const SPIRType &out_type, const SPIRType &in_type)
{
	// Ensure ISPC uses the (type) cast, rather than the type() cast.
	if (is_trivial_bitcast_glsl_op(out_type, in_type))
		return join("(" + type_to_glsl(out_type) + ")");

	return CompilerGLSL::bitcast_glsl_op(out_type, in_type);
}

void CompilerISPC::emit_resources()
{
	vector<string> varyings = { "varying", "uniform" };

	// Output all basic struct types which are not Block or BufferBlock as these are declared inplace
	// when such variables are instantiated.
	statement("");
	statement("//////////////////////////////");
	statement("// Resources");
	statement("//////////////////////////////");

	// Specialisation constants and constant ops before the resources, in case they are used within.
	emit_specialization_constants();

	for (auto &id : ids)
	{
		if (id.get_type() == TypeType)
		{
			auto &type = id.get<SPIRType>();
			if (type.basetype == SPIRType::Struct && type.array.empty() && !type.pointer &&
			    !(meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
			      meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock)))
			{
				emit_struct(type);
			}
		}
	}

	// Output UBOs and SSBOs
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && type.pointer && type.storage == StorageClassUniform &&
			    !is_hidden_variable(var) &&
			    (meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
			     meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock)))
			{
				emit_buffer_block(var);
			}
		}
	}

	// Output push constant blocks
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);
			if (!is_hidden_variable(var) && var.storage != StorageClassFunction && type.pointer &&
			    type.storage == StorageClassPushConstant)
			{
				emit_push_constant_block(var);
			}
		}
	}

	// Output in/out interfaces.
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && !is_hidden_variable(var) && type.pointer &&
			    (var.storage == StorageClassInput || var.storage == StorageClassOutput) &&
			    interface_variable_exists_in_entry_point(var.self))
			{
				emit_interface_block(var);
			}
		}
	}

	// Output Uniform Constants (values, samplers, images, etc).
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && !is_hidden_variable(var) && type.pointer &&
			    (type.storage == StorageClassUniformConstant ||
			     type.storage == StorageClassAtomicCounter)) // || type.storage == StorageClassPrivate))
			{
				emit_uniform(var);
			}
		}
	}

	// Emit regular globals which are allocated per invocation.
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (var.storage == StorageClassWorkgroup)
		{
			//			emit_shared(var);
		}
		else if (var.storage == StorageClassPrivate)
		{
			statement(CompilerGLSL::variable_decl(var), ";");
		}
	}

	statement("");

	statement("//////////////////////////////");
	statement("// Shader Code");
	statement("//////////////////////////////");
}

void CompilerISPC::emit_workgroup_variables()
{
	// Emit workgroup variables.
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (var.storage == StorageClassWorkgroup)
		{
			emit_shared(var);
		}
	}
}

// Start looking for the varying variables.
void CompilerISPC::find_vectorisation_variables()
{
	VectorisationHandler handler(*this);
	traverse_all_reachable_opcodes(get<SPIRFunction>(entry_point), handler);

	handler.propogate_ispc_varyings_for_builtins();
#if DUMP_VARYING_DEPENDANCIES
	handler.dump_varying_dependancies();
#endif
}

string CompilerISPC::compile()
{
	// Force a classic "C" locale, reverts when function returns
	ClassicLocale classic_locale;

	// Do not deal with ES-isms like precision, older extensions and such.
	options.es = false;
	options.version = 450;
	backend.float_literal_suffix = true;
	backend.double_literal_suffix = false;
	backend.long_long_literal_suffix = true;
	backend.uint32_t_literal_suffix = true;
	backend.basic_int_type = "int";
	backend.basic_uint_type = "int"; // We should support this at some point...
	backend.swizzle_is_function = false;
	backend.shared_is_implied = true;
	backend.flexible_member_array_supported = false;
	backend.explicit_struct_type = true;
	backend.use_initializer_list = true;
	backend.supports_native_swizzle = false;
	backend.stdlib_filename = "spirvcross_stdlib.ispc";

	update_active_builtins();

	// Write to the stdlib buffer, then swap the buffer pointers over.
	buffer = unique_ptr<ostringstream>(new ostringstream());
	emit_stdlib();
	stdlib_buffer = std::move(buffer);

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			SPIRV_CROSS_THROW("Over 3 compilation loops detected. Must be a bug!");

		reset();
		entry_point_ids.clear();

		// Convert the use of global variables to recursively-passed function parameters
		// Done in the compilation loop to ensure that any promoted runtimearray counts are correctly propogated
		localize_global_variables();
		extract_global_variables_from_functions();

		// Loop variables are special cases for vectorisation.
		find_loop_variables_from_functions();

		find_vectorisation_variables();

		buffer = unique_ptr<ostringstream>(new ostringstream());
		emit_header();
		emit_resources();

		emit_function(get<SPIRFunction>(entry_point), Bitset(0));

		pass_count++;
	} while (force_recompile);

	emit_ispc_main();

	return buffer->str();
}

void CompilerISPC::emit_ispc_main()
{
	auto &execution = get_entry_point();

	string entry_point_name = interface_name.empty() ? string("ispc") : interface_name;

	statement("");
	statement("//////////////////////////////");
	statement("// ISPC Entry Points");
	statement("//////////////////////////////");

	vector<string> tiled = { "", "_tiled" };

	for (auto &t : tiled)
	{

		// Dispatch all
		{
			string decl = "export void " + entry_point_name + "_dispatch_all" + t + "(uniform int work_groups[3]";
			string args = entry_point_args(!get<SPIRFunction>(entry_point).arguments.empty(), false, false);
			if (!args.empty())
				decl = join(decl, ", ", args);
			decl += ")";
			statement(decl);
		}

		{
			begin_scope();

			statement("uniform int3 gl_NumWorkGroups = int3(work_groups[0], work_groups[1], work_groups[2]);");
			statement("");

			statement("// Loop over the work group dimensions");
			statement("for(uniform int z = 0; z < gl_NumWorkGroups.z; z++)");
			{
				begin_scope();
				statement("for(uniform int y = 0; y < gl_NumWorkGroups.y; y++)");
				{
					begin_scope();
					statement("for(uniform int x = 0; x < gl_NumWorkGroups.x; x++)");
					{
						begin_scope();
						statement("uniform int3 gl_WorkGroupID = int3(x, y, z);"); //
						emit_workgroup_variables();

						statement("");
						statement("// Vectorise the workgroup");
						if (execution.workgroup_size.z > 1)
						{
							statement("foreach" + t +
							          "(lz = 0 ... gl_WorkGroupSize.z, ly = 0 ... gl_WorkGroupSize.y, "
							          "lx = 0 ... gl_WorkGroupSize.x)");
							begin_scope();
							statement("varying int3 gl_LocalInvocationID = int3(lx, ly, lz);");
						}
						else if (execution.workgroup_size.y > 1)
						{
							statement("foreach" + t + "(ly = 0 ... gl_WorkGroupSize.y, lx = 0 ... gl_WorkGroupSize.x)");
							begin_scope();
							statement("varying int3 gl_LocalInvocationID = int3(lx, ly, 0);");
						}
						else
						{
							statement("foreach" + t + "(lx = 0 ... gl_WorkGroupSize.x)");
							begin_scope();
							statement("varying int3 gl_LocalInvocationID = int3(lx, 0, 0);");
						}
						{
							statement("varying int3 gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + "
							          "gl_LocalInvocationID;");
							statement(
							    "varying int gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * "
							    "gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + "
							    "gl_LocalInvocationID.x;");
							statement("");

							string decl = entry_point_name;
							decl += "_ispc_main(";
							decl += entry_point_args_init(!get<SPIRFunction>(entry_point).arguments.empty());
							decl += ");";
							statement(decl);
							end_scope();
						}
						end_scope();
					}
					end_scope();
				}
				end_scope();
			}
			end_scope();
		}
		statement("");

		// Dispatch single
		{
			string decl = "export void " + entry_point_name + "_dispatch_single" + t +
			              "(uniform int work_group_ID[3], uniform int work_groups[3]";
			string args = entry_point_args(!get<SPIRFunction>(entry_point).arguments.empty(), false, false);
			if (!args.empty())
				decl = join(decl, ", ", args);
			decl += ")";
			statement(decl);
		}
		{
			begin_scope();
			statement("uniform int3 gl_NumWorkGroups = int3(work_groups[0], work_groups[1], work_groups[2]);");
			statement("uniform int3 gl_WorkGroupID = int3(work_group_ID[0], work_group_ID[1], work_group_ID[2]);");
			emit_workgroup_variables();
			statement("");
			statement("// Vectorise the workgroup");
			if (execution.workgroup_size.z > 1)
			{
				statement("foreach" + t +
				          "(lz = 0 ... gl_WorkGroupSize.z, ly = 0 ... gl_WorkGroupSize.y, "
				          "lx = 0 ... gl_WorkGroupSize.x)");
				begin_scope();
				statement("varying int3 gl_LocalInvocationID = int3(lx, ly, lz);");
			}
			else if (execution.workgroup_size.y > 1)
			{
				statement("foreach" + t + "(ly = 0 ... gl_WorkGroupSize.y, lx = 0 ... gl_WorkGroupSize.x)");
				begin_scope();
				statement("varying int3 gl_LocalInvocationID = int3(lx, ly, 0);");
			}
			else
			{
				statement("foreach" + t + "(lx = 0 ... gl_WorkGroupSize.x)");
				begin_scope();
				statement("varying int3 gl_LocalInvocationID = int3(lx, 0, 0);");
			}
			{
				statement(
				    "varying int3 gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID;");
				statement("varying int gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * "
				          "gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;");
				statement("");

				string decl = entry_point_name;
				decl += "_ispc_main(";
				decl += entry_point_args_init(!get<SPIRFunction>(entry_point).arguments.empty());
				decl += ");";
				statement(decl);

				end_scope();
			}
			end_scope();
		}
		statement("");
	}

	statement("");
	statement("export void ", entry_point_name,
	          "_get_workgroup_size(uniform int & wg_x, uniform int & wg_y, uniform int & wg_z)");
	begin_scope();
	statement("wg_x = gl_WorkGroupSize.x;");
	statement("wg_y = gl_WorkGroupSize.y;");
	statement("wg_z = gl_WorkGroupSize.z;");
	end_scope();
	statement("");
}

void CompilerISPC::emit_function_prototype(SPIRFunction &func, const Bitset &)
{
	local_variable_names = resource_names;
	string decl;

	auto &type = get<SPIRType>(func.return_type);
	decl += "static SPIRV_INLINE ";
	if (type.basetype != SPIRType::Void)
		decl += meta[func.self].decoration.ispc_varying ? "varying " : "uniform ";
	decl += type_to_glsl(type);
	decl += " ";

	if (func.self == entry_point)
	{
		string entry_point_name = interface_name.empty() ? string("ispc") : interface_name;
		decl += entry_point_name;
		decl += "_ispc_main";
		processing_entry_point = true;
	}
	else
		decl += to_name(func.self);

	decl += "(";

	if (processing_entry_point)
	{
		decl += entry_point_args(!func.arguments.empty(), true, true);
	}

	for (auto &arg : func.arguments)
	{
		//	add_local_variable_name(arg.id);

		decl += argument_decl(arg);
		if (&arg != &func.arguments.back())
			decl += ", ";

		// Hold a pointer to the parameter so we can invalidate the readonly field if needed.
		auto *var = maybe_get<SPIRVariable>(arg.id);
		if (var)
			var->parameter = &arg;
	}

	decl += ")";
	statement(decl);
}

bool CompilerISPC::optimize_read_modify_write(const SPIRType &type, const string &lhs, const string &rhs)
{
	// Do this with strings because we have a very clear pattern we can check for and it avoids
	// adding lots of special cases to the code emission.
	if (rhs.size() < lhs.size() + 3)
		return false;

	// Do not optimize matrices. They are a bit awkward to reason about in general
	// (in which order does operation happen?), and it does not work on MSL anyways.
	// ISPC - Reinstate if ISPC starts supporting += overloading
	//   if (type.vecsize > 1 && type.columns > 1)
	//       return false;

	// ISPC does not allow operater overloading for structs for +=, so don't allow optimisation for vectors
	if (type.vecsize > 1)
		return false;

	auto index = rhs.find(lhs);
	if (index != 0)
		return false;

	// TODO: Shift operators, but it's not important for now.
	auto op = rhs.find_first_of("+-/*%|&^", lhs.size() + 1);
	if (op != lhs.size() + 1)
		return false;

	// Check that the op is followed by space. This excludes && and ||.
	if (rhs[op + 1] != ' ')
		return false;

	char bop = rhs[op];
	auto expr = rhs.substr(lhs.size() + 3);
	// Try to find increments and decrements. Makes it look neater as += 1, -= 1 is fairly rare to see in real code.
	// Find some common patterns which are equivalent.
	if ((bop == '+' || bop == '-') && (expr == "1" || expr == "uint(1)" || expr == "1u" || expr == "int(1u)"))
		statement(lhs, bop, bop, ";");
	else
		statement(lhs, " ", bop, "= ", expr, ";");
	return true;
}

string CompilerISPC::argument_decl(const SPIRFunction::Parameter &arg)
{
	auto &type = expression_type(arg.id);
	bool constref = !type.pointer || arg.write_count == 0;

	// remove const for now as it causes some ISPC issues
	constref = false;

	auto &var = get<SPIRVariable>(arg.id);

	string base = type_to_glsl(type, arg.id);
	string variable_name = to_name(var.self);
	remap_variable_type_name(type, variable_name, base);

	for (uint32_t i = 0; i < type.array.size(); i++)
		variable_name = join(variable_name, "[", to_array_size(type, i), "]");

	// arrays get confused if passed by reference
	string passByRef = type.array.size() ? " " : "& ";

	return join(meta[arg.id].decoration.ispc_varying ? "varying " : "uniform ", constref ? "const " : "", base,
	            passByRef, variable_name);
}

string CompilerISPC::variable_decl(const SPIRType &type, const string &name, uint32_t id)
{
	string base;

	if (id > 0)
	{
		if (meta[id].decoration.ispc_varying)
			base = "varying ";
		else
			base = "uniform ";
	}
	else
	{
		// Should only get in here for struct members etc
	}

	base += type_to_glsl(type, id);
	remap_variable_type_name(type, name, base);
	bool runtime = false;
	string arrayString = "";

	for (uint32_t i = 0; i < type.array.size(); i++)
	{
		auto &array = type.array[i];
		if (!array && type.array_size_literal[i])
		{
			// Avoid using runtime arrays with std::array since this is undefined.
			// Runtime arrays cannot be passed around as values, so this is fine.
			runtime = true;
		}
		else
		{
			arrayString = join("[", to_array_size(type, i), "]");
		}
	}
	base += ' ';
	return base + name + (runtime ? "[]" : arrayString);
}

void CompilerISPC::emit_header()
{
	auto &execution = get_entry_point();
	std::tm localTime;
	std::time_t result = std::time(nullptr);

#ifdef _WIN32
	localtime_s(&localTime, &result);
#else
	localtime_r(&result, &localTime);
#endif

	statement("//");
	statement("//////////////////////////////");
	statement("// This ISPC kernel is autogenerated by spirv-cross.");
	statement("// ", std::put_time(&localTime, "%c"));
	statement("//////////////////////////////");
	statement("//");
	statement("");
	statement("#include \"", backend.stdlib_filename, "\"");

	statement("");
	statement("//////////////////////////////");
	statement("// Work Group");
	statement("//////////////////////////////");
	statement("static uniform int3 gl_WorkGroupSize = {", execution.workgroup_size.x, ", ", execution.workgroup_size.y,
	          ", ", execution.workgroup_size.z, "};");
	statement("");

	switch (execution.model)
	{
		//	case ExecutionModelGeometry:
		//	case ExecutionModelTessellationControl:
		//	case ExecutionModelTessellationEvaluation:
	case ExecutionModelGLCompute:
		//	case ExecutionModelFragment:
		//	case ExecutionModelVertex:
		//		statement("struct Shader");
		//		begin_scope();
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported execution model.");
	}

	switch (execution.model)
	{
	case ExecutionModelGeometry:
		impl_type = "GeometryShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "GeometryResources";
		break;

	case ExecutionModelVertex:
		impl_type = "VertexShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "VertexResources";
		break;

	case ExecutionModelFragment:
		impl_type = "FragmentShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "FragmentResources";
		break;

	case ExecutionModelGLCompute:
		impl_type = join("ComputeShader<Impl::Shader, Impl::Shader::Resources, ", execution.workgroup_size.x, ", ",
		                 execution.workgroup_size.y, ", ", execution.workgroup_size.z, ">");
		resource_type = "ComputeResources";
		break;

	case ExecutionModelTessellationControl:
		impl_type = "TessControlShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "TessControlResources";
		break;

	case ExecutionModelTessellationEvaluation:
		impl_type = "TessEvaluationShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "TessEvaluationResources";
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported execution model.");
	}
}

// The optional id parameter indicates the object whose type we are trying
// to find the description for. It is optional. Most type descriptions do not
// depend on a specific object's use of that type.
string CompilerISPC::type_to_glsl(const SPIRType &type, uint32_t id)
{
	// Ignore the pointer type since GLSL doesn't have pointers.

	switch (type.basetype)
	{
	case SPIRType::Struct:
		// Need OpName lookup here to get a "sensible" name for a struct.
		if (backend.explicit_struct_type)
			return join("struct ", to_name(type.self));
		else
			return to_name(type.self);

	case SPIRType::Image:
	case SPIRType::SampledImage:
		return image_type_glsl(type, id);

	case SPIRType::Sampler:
		// The depth field is set by calling code based on the variable ID of the sampler, effectively reintroducing
		// this distinction into the type system.
		return comparison_samplers.count(id) ? "samplerShadow" : "sampler";

	case SPIRType::Void:
		return "void";

	default:
		break;
	}

	if (type.vecsize == 1 && type.columns == 1) // Scalar builtin
	{
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return "bool";
		case SPIRType::Int:
			return backend.basic_int_type;
		case SPIRType::UInt:
			return backend.basic_uint_type;
		case SPIRType::AtomicCounter:
			return "atomic_uint";
		case SPIRType::Float:
			return "float";
		case SPIRType::Double:
			return "double";
		case SPIRType::Int64:
			return "int64";
		case SPIRType::UInt64:
			return "unsigned int64";
		default:
			return "???";
		}
	}
	else if (type.vecsize > 1 && type.columns == 1) // Vector builtin
	{
		// Its a struct, so can;t use short vectors if it is to be shared with the app
		if (id == 0)
		{
			switch (type.basetype)
			{
			case SPIRType::Boolean:
				return join("bool", type.vecsize);
			case SPIRType::Int:
				return join("int", type.vecsize);
			case SPIRType::UInt:
				return join("int", type.vecsize);
			case SPIRType::Float:
				return join("float", type.vecsize);
			case SPIRType::Double:
				return join("double", type.vecsize);
			case SPIRType::Int64:
				return join("int64", type.vecsize);
			case SPIRType::UInt64:
				return join("int64", type.vecsize);
			default:
				return "???";
			}
		}
		else
		{
			// Shouldn't be using short vectors due to padding issues. Should remove.
			switch (type.basetype)
			{
			case SPIRType::Boolean:
				return join("bool<", type.vecsize, ">");
			case SPIRType::Int:
				return join("int", type.vecsize);
			case SPIRType::UInt:
				return join("int", type.vecsize);
			case SPIRType::Float:
				return join("float", type.vecsize);
			case SPIRType::Double:
				return join("double<", type.vecsize, ">");
			case SPIRType::Int64:
				return join("int64<", type.vecsize, ">");
			case SPIRType::UInt64:
				return join("int64<", type.vecsize, ">");
			default:
				return "???";
			}
		}
	}
	else if (type.vecsize == type.columns) // Simple Matrix builtin
	{
		// Not fully supported yet. Will need stdlib implementations.
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bmat", type.vecsize);
		case SPIRType::Int:
			return join("imat", type.vecsize);
		case SPIRType::UInt:
			return join("umat", type.vecsize);
		case SPIRType::Float:
			return join("mat", type.vecsize);
		case SPIRType::Double:
			return join("dmat", type.vecsize);
			// Matrix types not supported for int64/uint64.
		default:
			return "???";
		}
	}
	else
	{
		// Not fully supported yet. Will need stdlib implementations.
		switch (type.basetype)
		{
		case SPIRType::Boolean:
			return join("bmat", type.columns, "x", type.vecsize);
		case SPIRType::Int:
			return join("imat", type.columns, "x", type.vecsize);
		case SPIRType::UInt:
			return join("umat", type.columns, "x", type.vecsize);
		case SPIRType::Float:
			return join("mat", type.columns, "x", type.vecsize);
		case SPIRType::Double:
			return join("dmat", type.columns, "x", type.vecsize);
			// Matrix types not supported for int64/uint64.
		default:
			return "???";
		}
	}
}

// Since ISPC does not allow arrays to be copied via simple variable assignment,
// if the LHS and RHS represent an assignment of an entire array, it must be
// implemented by calling an array copy function.
// Returns whether the struct assignment was emitted.
// This should be optimised if the ISPC compiler doesn't already optimise it.
bool CompilerISPC::maybe_emit_array_assignment(uint32_t id_lhs, uint32_t id_rhs)
{
	// Assignment from an array initializer is fine.
	if (ids[id_rhs].get_type() == TypeConstant)
		return false;

	// We only care about assignments of an entire array
	auto &type = expression_type(id_rhs);
	if (type.array.size() == 0)
		return false;

	// Ensure the LHS variable has been declared
	auto *p_v_lhs = maybe_get_backing_variable(id_lhs);
	if (p_v_lhs)
		flush_variable_declaration(p_v_lhs->self);

	statement("for (uniform int ii = 0; ii < ", to_array_size(type, 0), "; ii++)");
	begin_scope();
	statement(to_expression(id_lhs), "[ii] = ", to_expression(id_rhs), "[ii];");
	end_scope();
	register_write(id_lhs);

	return true;
}

void CompilerISPC::emit_instruction(const Instruction &instruction)
{
	auto ops = stream(instruction);
	auto opcode = static_cast<Op>(instruction.op);
	uint32_t length = instruction.length;

#define BOP(op) emit_binary_op(ops[0], ops[1], ops[2], ops[3], #op)
#define BOP_CAST(op, type) \
	emit_binary_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define UOP(op) emit_unary_op(ops[0], ops[1], ops[2], #op)
#define QFOP(op) emit_quaternary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], #op)
#define TFOP(op) emit_trinary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], #op)
#define BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define BFOP_CAST(op, type) \
	emit_binary_func_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define UFOP(op) emit_unary_func_op(ops[0], ops[1], ops[2], #op)

	switch (opcode)
	{
	case OpInBoundsAccessChain:
	case OpAccessChain:
	{
		auto *var = maybe_get<SPIRVariable>(ops[2]);
		if (var)
			flush_variable_declaration(var->self);

		// If the base is immutable, the access chain pointer must also be.
		// If an expression is mutable and forwardable, we speculate that it is immutable.
		bool need_transpose;
		string e;

		// If the index is a constant, then it could be a uniform or a varying constant, depending on the varying-ness of the access chain.
		// If it is a varying, then we simply pass in the varying constant instead of the uniform one.
		// Under the hood, this will be cast to a varying value
		if (meta[ops[1]].decoration.ispc_varying)
		{
			// search for a constant
			// The ops are const, so copy into a non const array to allow us to modify them.
			assert((length - 3) < 8);
			uint32_t ops3[8] = { 0 };
			uint32_t new_length = length - 3;
			for (uint32_t ii = 0; ii < new_length; ii++)
			{
				ops3[ii] = ops[3 + ii];
				auto c = maybe_get<SPIRConstant>(ops[3 + ii]);
				if (c && (c->varying_constant_id != -1))
					ops3[ii] = c->varying_constant_id;
				else
					ops3[ii] = ops[3 + ii];
			}
			e = access_chain(ops[2], ops3, new_length, get<SPIRType>(ops[0]), &need_transpose);
		}
		else
		{
			e = access_chain(ops[2], &ops[3], length - 3, get<SPIRType>(ops[0]), &need_transpose);
		}
		auto &expr = set<SPIRExpression>(ops[1], move(e), ops[0], should_forward(ops[2]));
		expr.loaded_from = ops[2];
		expr.need_transpose = need_transpose;

		// ISPC needs access chains to be flagged as mutable.
		// This will force SPIRV-Cross to assign access chains to temporary variables and not forward them.
		// This is required for the case of :
		//   varying float4 = float4(ubo.array[idx].pos, 1.0)
		// where idx is a varying and ubo.array[] is a user provided uniform pointer to a runtime array,
		// and float4(varying float3, uniform float) requires a varying float3 as input.
		// As ubo.array[idx].pos is not in contiguous memory, it is not a varying and can not be used directly.
		// ISPC will throw an error about lvalues.
		// TODO : only use if the expression involves a runtime array
		if (get<SPIRType>(ops[0]).array.empty())
			expr.immutable = false;
		break;
	}

	case OpDot:
	{
		CompilerGLSL::emit_instruction(instruction);
		break;
	}

	// ISPC doesn't like
	//     float(val),
	// it prefers
	//    (float)val
	case OpBitcast:
	case OpConvertFToU:
	case OpConvertFToS:
	case OpConvertSToF:
	case OpConvertUToF:
	case OpUConvert:
	case OpSConvert:
	case OpFConvert:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t arg = ops[2];

		// Bitcasts can be used for casting between an int and a uint.
		// bitcast_glsl_op() treats this as a simple cast of the form int(val),
		// but ISPC needs (int)val, so we go down the normal cast path for that.
		if ((opcode == OpBitcast) && !is_trivial_bitcast_glsl_op(get<SPIRType>(result_type), expression_type(arg)))
		{
			// Normal bit cast
			auto op = bitcast_glsl_op(get<SPIRType>(result_type), expression_type(arg));
			emit_unary_func_op(result_type, id, arg, op.c_str());
			break;
		}
		auto type = get<SPIRType>(result_type);
		auto func = type_to_glsl_constructor(type);
		bool forward = should_forward(ops[2]);
		emit_op(result_type, id, join(func.c_str(), "(", to_expression(ops[2]), ")"), forward);

		inherit_expression_dependencies(id, ops[2]);

		break;
	}

	// ISPC doesn't support shuffles on structs, so we need to force the shuffle
	// to use constructor style usage instead.
	case OpVectorShuffle:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t vec0 = ops[2];
		uint32_t vec1 = ops[3];
		const auto *elems = &ops[4];
		length -= 4;

		auto &type0 = expression_type(vec0);

		// Force ISPC to shuffle.
		bool shuffle = !backend.supports_native_swizzle;
		/*		
        for (uint32_t i = 0; i < length; i++)
			if (elems[i] >= type0.vecsize)
				shuffle = true;
*/
		string expr;
		bool trivial_forward;

		if (shuffle)
		{
			trivial_forward = !expression_is_forwarded(vec0) && !expression_is_forwarded(vec1);

			// Constructor style and shuffling from two different vectors.
			vector<string> args;
			for (uint32_t i = 0; i < length; i++)
			{
				if (elems[i] >= type0.vecsize)
					args.push_back(join(to_enclosed_expression(vec1), ".", index_to_swizzle(elems[i] - type0.vecsize)));
				else
					args.push_back(join(to_enclosed_expression(vec0), ".", index_to_swizzle(elems[i])));
			}
			expr += join(type_to_glsl_constructor(get<SPIRType>(result_type)), "(", merge(args), ")");
		}
		else
		{
			trivial_forward = !expression_is_forwarded(vec0);

			// We only source from first vector, so can use swizzle.
			expr += to_enclosed_expression(vec0);
			expr += ".";
			for (uint32_t i = 0; i < length; i++)
				expr += index_to_swizzle(elems[i]);
			if (backend.swizzle_is_function && length > 1)
				expr += "()";
		}

		// A shuffle is trivial in that it doesn't actually *do* anything.
		// We inherit the forwardedness from our arguments to avoid flushing out to temporaries when it's not really needed.
		emit_op(result_type, id, expr, should_forward(vec0) && should_forward(vec1), trivial_forward);
		break;
	}

	case OpAtomicIAdd:
	case OpAtomicISub:
	case OpAtomicSMin:
	case OpAtomicUMin:
	case OpAtomicSMax:
	case OpAtomicUMax:
	case OpAtomicAnd:
	case OpAtomicOr:
	case OpAtomicXor:
	case OpAtomicExchange:
	{
		if (check_atomic_image(ops[2]))
		{
			SPIRV_CROSS_THROW("Atomic images not supported for ISPC.");
		}

		string func;
		switch (opcode)
		{
		case OpAtomicIAdd:
			func = "atomic_add";
			break;
		case OpAtomicISub:
			func = "atomic_subtract";
			break;
		case OpAtomicSMin:
		case OpAtomicUMin:
			func = "atomic_min";
			break;
		case OpAtomicSMax:
		case OpAtomicUMax:
			func = "atomic_max";
			break;
		case OpAtomicAnd:
			func = "atomic_and";
			break;
		case OpAtomicOr:
			func = "atomic_or";
			break;
		case OpAtomicXor:
			func = "atomic_xor";
			break;
		case OpAtomicExchange:
			func = "atomic_swap";
			break;
		default:
			break;
		}

		forced_temporaries.insert(ops[1]);
		auto expr = join(func, "(&", to_expression(ops[2]), ", ", to_enclosed_expression(ops[5]), ")");
		emit_op(ops[0], ops[1], expr, should_forward(ops[2]) && should_forward(ops[5]));
		flush_all_atomic_capable_variables();
		register_read(ops[1], ops[2], should_forward(ops[2]));
		break;
	}

    case OpStore:
    {
        if (maybe_emit_array_assignment(ops[0], ops[1]))
            break;
#if 0
        auto &type = expression_type(ops[0]);

        // We're a matrix. Need to do something different as ISPC won't allow arrays to be copied
        if (type.columns != 1)
        {
            auto *p_lhs = maybe_get_backing_variable(ops[0]);
            if (p_lhs)
                flush_variable_declaration(p_lhs->self);

            auto lhs = to_expression(ops[0]);
            auto rhs = to_expression(ops[1]);
            statement("mat", type.columns, "_copy(", lhs, ", ", rhs, ");");
            break;
        }
#endif
        CompilerGLSL::emit_instruction(instruction);
        break;
    }

    case OpLoad:
    {
#if 0
        uint32_t result_type = ops[0];
        uint32_t id = ops[1];
        uint32_t ptr = ops[2];

        const auto ptr_type = expression_type(ptr);
        if (ptr_type.columns != 1)
        {
            flush_variable_declaration(ptr);
            auto *p_lhs = maybe_get_backing_variable(ops[0]);
            if (p_lhs)
                flush_variable_declaration(p_lhs->self);

            auto lhs = to_expression(id);


            // If an expression is mutable and forwardable, we speculate that it is immutable.
            bool forward = should_forward(ptr) && forced_temporaries.find(id) == end(forced_temporaries);

            // If loading a non-native row-major matrix, mark the expression as need_transpose.
            bool need_transpose = false;
            bool old_need_transpose = false;

            auto *ptr_expression = maybe_get<SPIRExpression>(ptr);
            if (ptr_expression && ptr_expression->need_transpose)
            {
                old_need_transpose = true;
                ptr_expression->need_transpose = false;
                need_transpose = true;
            }
            else if (is_non_native_row_major_matrix(ptr))
                need_transpose = true;

            auto rhs = to_expression(ptr);

            if (ptr_expression)
                ptr_expression->need_transpose = old_need_transpose;

            // We want to declare the temporary, but not initialise it.
            // Standard function always puts an assignment operator there
            // So call the function so it does the right thing with hoisting any temps etc
            // then just declare it locally.
            auto temp = declare_temporary(result_type, id);
            auto &type = get<SPIRType>(result_type);
            statement(variable_decl(type, to_name(id), id), ";");
            statement("mat", ptr_type.columns, "_copy(", lhs, ", ", rhs, ");");

            auto expr = set<SPIRExpression>(id, to_name(id), result_type, true);
            expr.need_transpose = need_transpose;
            register_read(id, ptr, forward);

            // Pass through whether the result is of a packed type.
            if (has_decoration(ptr, DecorationCPacked))
                set_decoration(id, DecorationCPacked);

            break;
        }
#endif
        CompilerGLSL::emit_instruction(instruction);
        break;
    }

	// ISPC only sees a pointer for runtime arrays, so the array length needs passing in as 'meta' data.
	// This is currently done by creating a new global input variable that is passed in via the API.
	case OpArrayLength:
	{
		uint32_t result_type = ops[0];
		uint32_t id1 = ops[1];
		auto e = access_chain_internal(ops[2], &ops[3], length - 3, true);

		string name = ensure_valid_identifier(e + "_runtime_array_length", false);

		int32_t runtime_array_length_id = -1;
		uint32_t runtime_array_length_type_id = 0;

		// search to see if the runtime array length exists
		// search backwards for efficiency
		for (int target = ids.size() - 1; target >= 0; target--)
		{
			auto &id = ids.at(target);
			if (id.get_type() == TypeVariable)
			{
				auto &var = id.get<SPIRVariable>();
				if (get_name(var.self) == name)
				{
					runtime_array_length_id = var.self;
					break;
				}
			}
		}

		// If it doesn't exist, create it along with a new type
		if (runtime_array_length_id < 0)
		{
			uint32_t id = increase_bound_by(2);
			runtime_array_length_type_id = id + 0;

			auto &type = set<SPIRType>(runtime_array_length_type_id);

			type.basetype = SPIRType::UInt;
			type.pointer = false;
			type.storage = StorageClassInput;

			runtime_array_length_id = id + 1;
			set<SPIRVariable>(runtime_array_length_id, runtime_array_length_type_id, StorageClassInput);
			set_name(runtime_array_length_id, name);

			// force a recompile to ensure it is picked up as a global and pushed through the relevant entrypoints
			force_recompile = true;
		}

		set<SPIRExpression>(id1, name, result_type, true);
		// Store the ID as decoration so we can pick it up for the entry_point propogation
		meta[id1].decoration.runtime_array_length_id = runtime_array_length_id;
		break;
	}
	case OpControlBarrier:
	case OpMemoryBarrier:
	{
		if (!ignore_group_barriers)
			SPIRV_CROSS_THROW("Barriers are not currently supported by the ISPC backend. Use the option "
			                  "--ispc-ignore-group-barriers to ignore.");
		else
			statement("// barrier(); // Not currently supported");
		break;
	}

	default:
		CompilerGLSL::emit_instruction(instruction);
		break;
	}


}

string CompilerISPC::type_to_glsl_constructor(const SPIRType &type)
{
	if (type.array.size() > 1)
	{
		if (options.flatten_multidimensional_arrays)
			SPIRV_CROSS_THROW("Cannot flatten constructors of multidimensional array constructors, e.g. float[][]().");
		else if (!options.es && options.version < 430)
			require_extension("GL_ARB_arrays_of_arrays");
		else if (options.es && options.version < 310)
			SPIRV_CROSS_THROW("Arrays of arrays not supported before ESSL version 310.");
	}

	if (type.vecsize > 1 && type.columns == 1) // Scalar builtin
	{
		if (type.basetype == SPIRType::Float)
			return join("float", type.vecsize);
		if (type.basetype == SPIRType::Int)
			return join("int", type.vecsize);
		if (type.basetype == SPIRType::UInt)
			return join("int", type.vecsize);
	}
	else if (type.vecsize == 1 && type.columns == 1)
	{
		if (type.basetype == SPIRType::Float)
			return join("float_cast");
		if (type.basetype == SPIRType::Int)
			return join("int_cast");
		if (type.basetype == SPIRType::UInt)
			return join("int_cast");
	}

	auto e = type_to_glsl(type);
	for (uint32_t i = 0; i < type.array.size(); i++)
		e += "[]";
	return e;
}

void CompilerISPC::emit_glsl_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args, uint32_t count)
{
	GLSLstd450 op = static_cast<GLSLstd450>(eop);

	switch (op)
	{
		// Vector math
	case GLSLstd450Length:
		emit_unary_func_op(result_type, id, args[0], "length");
		break;

	case GLSLstd450Reflect:
		emit_binary_func_op(result_type, id, args[0], args[1], "reflect");
		break;

	case GLSLstd450FMix:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "mix");
		break;

	case GLSLstd450RoundEven:
		emit_unary_func_op(result_type, id, args[0], "round");
		break;

	case GLSLstd450InverseSqrt:
		emit_unary_func_op(result_type, id, args[0], "rsqrt");
		break;

	default:
		CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
	}
}

//
// This is the handler function for creating the dependancy tree used for figuring out the
// uniform/varyings.
//
bool CompilerISPC::VectorisationHandler::handle(spv::Op opcode, const uint32_t *args, uint32_t length)
{
	// dst = src; dst is dependant upon src, so [src] = dst
	auto add_dependancies = [&](const uint32_t dst, const uint32_t src) {
		dependee_hierarchy[src].insert(dst);
		dependant_hierarchy[dst].insert(src);
	};

	switch (opcode)
	{
	case OpAccessChain:
	case OpInBoundsAccessChain:
	{
		if (length < 3)
			return false;

		// The access chain is name, type, arg indices. Need to add all dependancies
		for (uint32_t i = 2; i < length; i++)
		{
			add_dependancies(args[1], args[i]);

			// Access chains need to be 2-way as they are simply indirections.
			// But, if the src is a global passed in by the user, then we don't as they are always uniform
			// but perhaps with varying runtime arrays.
			{
				auto *var = compiler.maybe_get<SPIRVariable>(args[i]);
				if (var && var->storage == StorageClassFunction)
				{
					add_dependancies(args[i], args[1]);
				}
			}

			// A constant index may need flagging as a varying
			{
				auto *var = compiler.maybe_get<SPIRConstant>(args[i]);
				if (var)
				{
					if (var->varying_constant_id == -1)
					{
						// Create a new ID which is a varying and assign it here
						uint32_t next_id = compiler.increase_bound_by(1);
						compiler.set<SPIRConstant>(next_id, var->constant_type, var->m.c[0].r[0].u32,
						                           var->specialization);
						var->varying_constant_id = next_id;

						// Ensure the existing variable has all the same meta info and is flagged as a varying
						compiler.meta[next_id] = compiler.meta[args[i]];
						compiler.meta[next_id].decoration.ispc_varying = true;
					}
				}
			}
		}

		break;
	}

	case OpStore:
	{
		if (length < 2)
			return false;

		add_dependancies(args[0], args[1]);

		auto *var = compiler.maybe_get<SPIRVariable>(args[0]);

		// Ignore loop variables as they should always be uniform
		if (var && !var->loop_variable)
		{
			//
			// If the variable has scope outside of the blocks contained within the current conditional,
			// add it as a dependancy on the current conditional.
			// Walk the stack of conditionals if true.
			// ie.
			// float a = 0.0;
			// if (b)
			//    a = 1.0;
			//
			// if b is a varying conditional, then a also needs to be a varying variable
			//
			unordered_set<uint32_t> visited_blocks;

			// The stack of nested conditional blocks is updated by the handler whenever a new conditional block is entered.
			// It therefore contains a stack of dependant conditional blocks that all depend on each other, so if a
			// high level conditional block is a varying, then all of its child conditional blocks will also be varyings.
			//
			// If a store occurs within a (possibly nested) conditional block, then the variable should depend on the conditional.
			// The exceptions are :
			//   * If the conditional block is in a different function
			//   * If store is within the conditionals this_block (which means the condition hasn't been branched yet)
			//   * If the variable is declared within the scope of the current conditional, then it shouldn't depend on
			//       the conditional as its not enough information to determine if it is a varying/uniform.
			for (auto conditional : condition_block_stack)
			{
				bool depend_on_condition = true;

				// Only walk the dependancy if the conditional is within the current function
				if (conditional->func != current_func)
					depend_on_condition = false;
				else
				{
					// Use the control flow graph that the handler generates per function.
					visited_blocks.clear();
					cfg_stack.top()->walk_from(visited_blocks, current_block->self, [&](uint32_t walk_block) {
						if (walk_block == conditional->this_block)
							depend_on_condition = false;
						else
						{
							auto &b = compiler.get<SPIRBlock>(walk_block);
							for (auto &v : b.dominated_variables)
							{

								// Found between current block and conditional, so no dependancy
								if (v == args[0])
									depend_on_condition = false;
							}
						}
					});
				}
				// Add the dependancy to the conditional
				if (depend_on_condition)
				{
					add_dependancies(args[0], conditional->condition);
				}
			}
		}
		break;
	}
	//    case OpFunctionParameter:
	case OpFunctionCall:
	{
		if (length < 3)
			return false;

		auto &func = compiler.get<SPIRFunction>(args[2]);
		const auto *arg = &args[3];
		length -= 3;

		// for functions, we need to add 2-way dependancies to ensure any
		// parameters promoted to varying within the function
		// are pushed back out of it.
		for (uint32_t i = 0; i < length; i++)
		{
			auto &argument = func.arguments[i];
			add_dependancies(argument.id, arg[i]);
			add_dependancies(arg[i], argument.id);
		}

		// Need to add a dependancy on the function return variable and the return of the last block in the function
		// This ensures that any variables returned and stored have their varying status correctly propogated
		// args[0] stores the return type
		// args[1] stores the return id

		// walk all blocks looking for return types and propogate to the function
		for (auto &b : func.blocks)
		{
			auto block = compiler.get<SPIRBlock>(b);
			if (block.return_value > 0)
			{
				// OpReturnValue can return Undef, so don't add a dependancy for this
				if (compiler.ids.at(block.return_value).get_type() != TypeUndef)
				{
					// Add dependancy for the func as well, as this is used when checking for the function signature return type
					add_dependancies(func.self, block.return_value);
					add_dependancies(args[1], block.return_value);
				}
			}
		}
		break;
	}

	case OpAtomicIAdd:
	case OpAtomicISub:
	case OpAtomicSMin:
	case OpAtomicUMin:
	case OpAtomicSMax:
	case OpAtomicUMax:
	case OpAtomicAnd:
	case OpAtomicOr:
	case OpAtomicXor:
	case OpAtomicExchange:
	{
		// Atomics take uniform arguments (a buffer), but return varying results, so force them to be varying
		compiler.meta[args[1]].decoration.ispc_varying = true;
		break;
	}

	default:
	{
		for (uint32_t i = 2; i < length; i++)
		{
			add_dependancies(args[1], args[i]);
		}
		break;
	}
	}
	return true;
}

void CompilerISPC::VectorisationHandler::set_current_function(const SPIRFunction &func)
{
	// Create the CFG for the current function and push it onto the stack
	cfg_stack.push(new CFG(compiler, func));
	current_func = &cfg_stack.top()->get_function();
}

bool CompilerISPC::VectorisationHandler::begin_function_scope(const uint32_t *, uint32_t)
{
	return true;
}

bool CompilerISPC::VectorisationHandler::end_function_scope(const uint32_t *, uint32_t)
{
	// Pop the stack
	CFG *cfg = cfg_stack.top();
	if (cfg)
		delete cfg;
	cfg_stack.pop();

	current_func = &cfg_stack.top()->get_function();

	return true;
}

void CompilerISPC::VectorisationHandler::set_current_block(const SPIRBlock &block)
{
	current_block = &block;

	// Add a new block to the conditional block tracker. This is required for the correct dependancy tracking
	// of stores to dependancy blocks.
	if (current_block->condition)
	{
		conditional_block_tracker *cbt = new conditional_block_tracker;
		cbt->condition = current_block->condition;
		cbt->this_block = current_block->self;
		cbt->next_block = current_block->next_block;
		cbt->func = current_func;

		if (!condition_block_stack.empty() && condition_block_stack.back()->func == current_func)
		{
			// Add a dependancy between multiple conditionals in the stack
			// This will ensure that an outer conditional branch will still affect any stores
			// that occur within the nested branches
			dependee_hierarchy[condition_block_stack.back()->condition].insert(cbt->condition);
		}

		condition_block_stack.push_back(cbt);
	}
	else
	{
		// Need to work out when we have left the conditional block so we can pop the cbt from the stack
		while (!condition_block_stack.empty())
		{
			conditional_block_tracker *cbt = condition_block_stack.back();

			bool inside_conditional = false;
			unordered_set<uint32_t> visited_blocks;
			visited_blocks.clear();

			// walk from the top condition block and see if we see the current block. If we don't, then we have left the conditional block, so pop it.
			cfg_stack.top()->walk_from(visited_blocks, cbt->this_block, [&](uint32_t walk_block) {
				if (walk_block == current_block->self)
					inside_conditional = true;
			});

			// Next block should point to the block that is executed once all conditional branches have executed
			// So this is the point that we remove the cbt from the stack
			if (!inside_conditional)
			{
				condition_block_stack.pop_back();
				delete cbt;
				cbt = nullptr;
			}
			else
			{
				break;
			}
		}
	}

	// We track the Phi variables here and add the appropriate dependancies
	for (auto &phi : block.phi_variables)
	{
		dependee_hierarchy[phi.local_variable].insert(phi.function_variable);
	}

	// We need to ensure that the current return value depends on the current branch condition.
	// This will ensure that a divergent branch that returns true/false will correctly pick up the correct varying
	// type for the function signature
	if (!condition_block_stack.empty() && condition_block_stack.back()->func == current_func)
	{
		// Don't add the dependancy if we are still on the conditional block because
		// we haven't branched yet to the inner blocks
		if (current_block->self != condition_block_stack.back()->this_block)
		{
			if (current_block->return_value > 0)
			{
				// OpReturnValue can return Undef, so don't add a dependancy for this
				if (compiler.ids.at(current_block->return_value).get_type() != TypeUndef)
				{
					dependee_hierarchy[condition_block_stack.back()->condition].insert(current_block->return_value);
				}
			}
		}
	}
}

// The dependacy graph has already been created at this point and some initial varying variable have already
// been seeded, so find the varyings and propogate down the dependancy tree
bool CompilerISPC::VectorisationHandler::propogate_ispc_varyings(const uint32_t var)
{
	bool bPropogate = false;
	auto varIt = dependee_hierarchy.find(var);
	if (varIt != dependee_hierarchy.end())
	{
		if (compiler.meta[var].decoration.ispc_varying)
		{
			// walk all variables that depend on var
			for (auto &dependantVarIt : dependee_hierarchy[var])
			{
				if (!compiler.meta[dependantVarIt].decoration.ispc_varying)
				{
					compiler.meta[dependantVarIt].decoration.ispc_varying = true;
					bPropogate = true;

					// Propogate to all variables dependant upon this variable
					propogate_ispc_varyings(dependantVarIt);
				}
			}
		}
	}

	return bPropogate;
}

// Currently not used, but saved for experimentation.
// Reverse propogation ended up converting too many variables to varying, but could be useful
// if we find a very tricky case that can't be solved any other way!
bool CompilerISPC::VectorisationHandler::reverse_propogate_ispc_varyings(const uint32_t var)
{
	bool bPropogate = false;
	auto varIt = dependant_hierarchy.find(var);
	if (varIt != dependant_hierarchy.end())
	{
		if (compiler.meta[var].decoration.ispc_varying)
		{
			// walk all variables that depend on var
			for (auto &dependantVarIt : dependant_hierarchy[var])
			{
				if (!compiler.meta[dependantVarIt].decoration.ispc_varying)
				{
					compiler.meta[dependantVarIt].decoration.ispc_varying = true;
					bPropogate = true;

					// Propogate to all variables dependant upon this variable
					reverse_propogate_ispc_varyings(dependantVarIt);
				}
			}
		}
	}

	return bPropogate;
}

// Seed the varyings here based on usage of the builtins. If the shader is using a local invocation ID, it
// is requesting vectorisation as the ID will be unique per SIMD lane.
bool CompilerISPC::VectorisationHandler::propogate_ispc_varyings_for_builtins()
{
	bool bPropogate = false;
	for (auto &dependeeVarIt : dependee_hierarchy)
	{
		auto *dependeeVar = compiler.maybe_get<SPIRVariable>(dependeeVarIt.first);

		// Start with the builtins that we know need to be varyings.
		if (dependeeVar && compiler.is_builtin_variable(*dependeeVar))
		{
			switch (compiler.meta[dependeeVar->self].decoration.builtin_type)
			{
			case BuiltInLocalInvocationId:
			case BuiltInGlobalInvocationId:
			case BuiltInLocalInvocationIndex:
				compiler.meta[dependeeVarIt.first].decoration.ispc_varying = true;
				bPropogate |= propogate_ispc_varyings(dependeeVarIt.first);
				break;
			default:
				break;
			}
		}
		// This will catch any pre-determined varying variables (like atomics).
		else if (compiler.meta[dependeeVarIt.first].decoration.ispc_varying)
		{
			bPropogate |= propogate_ispc_varyings(dependeeVarIt.first);
		}
	}
#if 0
    // walk the reverse propogation list
    for (auto &reverse_prop : reverse_propogation_varyings)
    {
        if (compiler.meta[reverse_prop].decoration.ispc_varying)
        {
            for (auto &dependandVarIt : dependant_hierarchy)
            {
                bPropogate |= reverse_propogate_ispc_varyings(dependandVarIt.first);
            }
        }
    }
#endif
	return bPropogate;
}

// Useful print mechanism to track varying dependancies
void CompilerISPC::VectorisationHandler::dump_varying_dependancy_branch(const uint32_t index, uint32_t tab_count,
                                                                        bool recurse,
                                                                        std::unordered_set<uint32_t> &branch)
{
	for (auto &dependantVarIt : dependee_hierarchy[index])
	{
		if (branch.count(dependantVarIt) == 0)
		{
			for (uint32_t tab = 0; tab < tab_count; tab++)
				fprintf(stdout, "-");
			fprintf(stdout, "> ");

			fprintf(stdout, "%s [%d] : %d\n", compiler.to_name(dependantVarIt).c_str(), dependantVarIt,
			        compiler.meta[dependantVarIt].decoration.ispc_varying);

			if (recurse)
			{
				branch.insert(dependantVarIt);
				dump_varying_dependancy_branch(dependantVarIt, tab_count + 1, recurse, branch);
				//                branch.erase(dependantVarIt);
			}
		}
	}
};

void CompilerISPC::VectorisationHandler::dump_varying_dependancies()
{
	fprintf(stdout, "============\n\n");
	fprintf(stdout, "\n\nVarying Dependancies\n\n");

	std::unordered_set<uint32_t> branch;

	// varyings first
	for (auto &dependeeVarIt : dependee_hierarchy)
	{
		if (compiler.meta[dependeeVarIt.first].decoration.ispc_varying)
		{
			if (branch.count(dependeeVarIt.first) == 0)
			{
				fprintf(stdout, "%s [%d] : %d\n", compiler.to_name(dependeeVarIt.first).c_str(), dependeeVarIt.first,
				        compiler.meta[dependeeVarIt.first].decoration.ispc_varying);
				branch.insert(dependeeVarIt.first);
				dump_varying_dependancy_branch(dependeeVarIt.first, 2, true, branch);
				//            branch.erase(dependeeVarIt.first);
				fprintf(stdout, "\n");
			}
		}
	}

	branch.clear();

	// then uniforms
	fprintf(stdout, "\n\nUniform Dependancies\n\n");
	for (auto &dependeeVarIt : dependee_hierarchy)
	{
		if (!compiler.meta[dependeeVarIt.first].decoration.ispc_varying)
		{
			if (branch.count(dependeeVarIt.first) == 0)
			{
				fprintf(stdout, "%s [%d] : %d\n", compiler.to_name(dependeeVarIt.first).c_str(), dependeeVarIt.first,
				        compiler.meta[dependeeVarIt.first].decoration.ispc_varying);

				branch.insert(dependeeVarIt.first);
				dump_varying_dependancy_branch(dependeeVarIt.first, 2, true, branch);
				//                branch.erase(dependeeVarIt.first);
				fprintf(stdout, "\n");
			}
		}
	}
	fprintf(stdout, "============\n\n");
}

void CompilerISPC::find_loop_variables_from_functions()
{
	unordered_set<uint32_t> processed_functions;

	find_loop_variables_from_function(entry_point, processed_functions);
}

// Find loop variables within a function as they shouldn't depend on any conditionals.
void CompilerISPC::find_loop_variables_from_function(uint32_t func_id,
                                                     std::unordered_set<uint32_t> &processed_functions)
{
	// don't process a func multiple times
	if (processed_functions.count(func_id) > 0)
		return;

	processed_functions.insert(func_id);

	auto &func = get<SPIRFunction>(func_id);

	// This *may* have side affects, but it should work out the loop variables.
	// These are needed as we don't want to make these dependant on conditional branches
	// and they are represented firstly by a store (initialise), followed by a loop construct.
	// Otherwise, when doing the varying analysis, we don't know if it is a loop variable.
	if (!func.analyzed_variable_scope)
		analyze_variable_scope(func);

	for (auto block : func.blocks)
	{
		auto &b = get<SPIRBlock>(block);
		for (auto &i : b.ops)
		{
			auto ops = stream(i);
			auto op = static_cast<Op>(i.op);

			switch (op)
			{
			case OpFunctionCall:
			{
				uint32_t inner_func_id = ops[2];
				find_loop_variables_from_function(inner_func_id, processed_functions);
				break;
			}
			default:
				break;
			}
		}

		// *A* side affect of calling analyze_variable_scope is that the loop variable list is populated
		// This can cause variables in for loop init lists to be declared twice.
		// Same is true for temporaries
		// Need a better solution to this... :(
		if (!func.analyzed_variable_scope)
		{
			b.loop_variables.clear();
			b.declare_temporary.clear();
		}
	}
}

// For any global variable accessed directly by a function,
// extract that variable and add it as an argument to that function.
void CompilerISPC::extract_global_variables_from_functions()
{

	// Uniforms
	unordered_set<uint32_t> global_var_ids;
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			if (var.storage == StorageClassInput || var.storage == StorageClassUniform ||
			    var.storage == StorageClassUniformConstant || var.storage == StorageClassPushConstant)
			//            if (var.storage == StorageClassUniform)
			{
				global_var_ids.insert(var.self);
			}
		}
	}

	// Local vars that are declared in the main function and accessed directy by a function
	auto &entry_func = get<SPIRFunction>(entry_point);
	for (auto &var : entry_func.local_variables)
		global_var_ids.insert(var);

	std::set<uint32_t> added_arg_ids;
	unordered_set<uint32_t> processed_func_ids;
	extract_global_variables_from_function(entry_point, added_arg_ids, global_var_ids, processed_func_ids);
}

// ISPC does not support the use of global variables for shader input content.
// For any global variable accessed directly by the specified function, extract that variable,
// add it as an argument to that function, and the arg to the added_arg_ids collection.
void CompilerISPC::extract_global_variables_from_function(uint32_t func_id, std::set<uint32_t> &added_arg_ids,
                                                          unordered_set<uint32_t> &global_var_ids,
                                                          unordered_set<uint32_t> &processed_func_ids)
{
	// Avoid processing a function more than once
	if (processed_func_ids.find(func_id) != processed_func_ids.end())
	{
		// Return function global variables
		added_arg_ids = function_global_vars[func_id];
		return;
	}

	processed_func_ids.insert(func_id);

	auto &func = get<SPIRFunction>(func_id);

	// Function arguments get processed on each pass of the compiler due to the dynamic runtime array length variables
	// As such, clear the arg list of any globals before starting the processing.
	func.clear_aliased_global_variable_parameters();

	// Recursively establish global args added to functions on which we depend.
	for (auto block : func.blocks)
	{
		auto &b = get<SPIRBlock>(block);
		for (auto &i : b.ops)
		{
			auto ops = stream(i);
			auto op = static_cast<Op>(i.op);

			switch (op)
			{
			case OpLoad:
			case OpAccessChain:
			{
				uint32_t base_id = ops[2];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);

				break;
			}
			case OpFunctionCall:
			{
				uint32_t inner_func_id = ops[2];
				std::set<uint32_t> inner_func_args;
				extract_global_variables_from_function(inner_func_id, inner_func_args, global_var_ids,
				                                       processed_func_ids);
				added_arg_ids.insert(inner_func_args.begin(), inner_func_args.end());
				break;
			}
			// Array lengths are special.
			// In ISPC, there is no way to determine the length of a runtime array from a pointer alone, so the length is
			// passed in as a variable. This check ensures it is propogated to all of the relevant function entry points.
			case OpArrayLength:
			{
				uint32_t id = ops[1];
				uint32_t runtime_array_length_id = meta[id].decoration.runtime_array_length_id;
				if (runtime_array_length_id != 0)
					if (global_var_ids.find(runtime_array_length_id) != global_var_ids.end())
						added_arg_ids.insert(runtime_array_length_id);
			}

			default:
				break;
			}
		}
	}

	function_global_vars[func_id] = added_arg_ids;

	// Add the global variables as arguments to the function
	if (func_id != entry_point)
	{
		uint32_t next_id = increase_bound_by(uint32_t(added_arg_ids.size()));
		for (uint32_t arg_id : added_arg_ids)
		{
			uint32_t type_id = get<SPIRVariable>(arg_id).basetype;
			func.add_parameter(type_id, next_id, true);
			set<SPIRVariable>(next_id, type_id, StorageClassFunction);

			// Ensure the existing variable has a valid name and the new variable has all the same meta info
			set_name(arg_id, ensure_valid_name(to_name(arg_id), "v"));
			meta[next_id] = meta[arg_id];

			next_id++;
		}
	}
}

// Move the Private global variables to the entry function.
// Non-constant variables cannot have global scope in Metal.
void CompilerISPC::localize_global_variables()
{
	auto &entry_func = get<SPIRFunction>(entry_point);
	auto iter = global_variables.begin();
	while (iter != global_variables.end())
	{
		uint32_t gv_id = *iter;
		auto &gbl_var = get<SPIRVariable>(gv_id);
		if (gbl_var.storage == StorageClassPrivate)
		{
			gbl_var.storage = StorageClassFunction;
			entry_func.add_local_variable(gv_id);
			iter = global_variables.erase(iter);
		}
		else
			iter++;
	}
}

// Ensures that the specified name is permanently usable by prepending a prefix
// if the first chars are _ and a digit, which indicate a transient name.
string CompilerISPC::ensure_valid_name(string name, string pfx)
{
	if (name.size() >= 2 && name[0] == '_' && isdigit(name[1]))
	{
		return join(pfx, name);
	}
	else
	{
		return name;
	}
}

void CompilerISPC::find_entry_point_args()
{
	if (!entry_point_ids.empty())
		return;

	// Uniforms
	for (auto &id : ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
			     var.storage == StorageClassPushConstant) &&
			    !is_hidden_variable(var))
			{
				switch (type.basetype)
				{
				case SPIRType::Struct:
				{
					auto &m = meta.at(type.self);
					if (m.members.size() == 0)
						break;

					entry_point_ids.push_back(&id);
					break;
				}
				case SPIRType::Image:
				{
					entry_point_ids.push_back(&id);
					break;
				}
				default:
					break;
				}
			}
			if (var.storage == StorageClassInput || var.storage == StorageClassWorkgroup)
			{
				entry_point_ids.push_back(&id);
			}
		}
	}

	// Sort the args based on their names. This ensures that the ordering is consistant between
	// multiple runs of the compiler.
	if (entry_point_ids.size() > 1)
		sort(entry_point_ids.begin(), entry_point_ids.end(), [=](const Variant *a, const Variant *b) -> bool {
			auto &varA = a->get<SPIRVariable>();
			auto &varB = b->get<SPIRVariable>();

			auto nameA = to_name(varA.self);
			auto nameB = to_name(varB.self);

			return nameA > nameB;
		});
}

// Returns a string containing a comma-delimited list of args for the entry point function
string CompilerISPC::entry_point_args(bool append_comma, bool want_builtins, bool want_workgroup_vars)
{
	find_entry_point_args();

	string ep_args;

	// Uniforms
	for (auto &id : entry_point_ids)
	{
		auto &var = id->get<SPIRVariable>();
		auto &type = get<SPIRType>(var.basetype);

		uint32_t var_id = var.self;

		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassPushConstant) &&
		    !is_hidden_variable(var))
		{
			switch (type.basetype)
			{
			case SPIRType::Struct:
			{
				string varying = meta[var_id].decoration.ispc_varying ? "varying " : "uniform ";
				auto &m = meta.at(type.self);
				if (m.members.size() == 0)
					break;
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += " " + varying + type_to_glsl(type, var_id) + "& " + to_name(var_id);
				break;
			}
			case SPIRType::Image:
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += "uniform " + type_to_glsl(type, var_id) + " &" + to_name(var_id);
				break;
			default:
				break;
			}
		}
		bool workgroup_var = (var.storage == StorageClassWorkgroup);
		if (var.storage == StorageClassInput || workgroup_var)
		{
			if (is_builtin_variable(var) && want_builtins)
			{
				// Found a case where an unused builtin hadn't been flagged as varying because it wasn't used/processed.
				// This should be a a catchall.
				switch (meta[var.self].decoration.builtin_type)
				{
				case BuiltInLocalInvocationId:
				case BuiltInGlobalInvocationId:
				case BuiltInLocalInvocationIndex:
					meta[var_id].decoration.ispc_varying = true;
					break;
				default:
					break;
				}
			}

			if (workgroup_var)
			{
				meta[var_id].decoration.ispc_varying = false;
			}

			bool add_arg = true;
			if (is_builtin_variable(var) && !want_builtins)
				add_arg = false;
			if (workgroup_var && !want_workgroup_vars)
				add_arg = false;

			if (add_arg)
			{
				string varying = meta[var_id].decoration.ispc_varying ? "varying " : "uniform ";
				if (!ep_args.empty())
					ep_args += ", ";

				// Need to ensure that workgroup arrays are passed as pointers/references.
				// TODO : Ensure these are correctly passed to other functions if required...
				if (is_array(type))
					ep_args += varying + type_to_glsl(type, var_id) + "* " + to_expression(var_id);
				else if (workgroup_var) // Need passing by reference
					ep_args += varying + type_to_glsl(type, var_id) + "& " + to_expression(var_id);
				else
					ep_args += varying + type_to_glsl(type, var_id) + " " + to_expression(var_id);
			}
		}
	}

	if (!ep_args.empty() && append_comma)
		ep_args += ", ";

	return ep_args;
}

string CompilerISPC::entry_point_args_init(bool append_comma)
{
	find_entry_point_args();

	string ep_args;
	// Uniforms
	for (auto &id : entry_point_ids)
	{
		auto &var = id->get<SPIRVariable>();
		auto &type = get<SPIRType>(var.basetype);

		uint32_t var_id = var.self;

		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassPushConstant) &&
		    !is_hidden_variable(var))
		{
			switch (type.basetype)
			{
			case SPIRType::Struct:
			{
				auto &m = meta.at(type.self);
				if (m.members.size() == 0)
					break;
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += " " + to_name(var_id);
				break;
			}
			case SPIRType::Image:
			{
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += " " + to_name(var_id);
				break;
			}
			default:
				break;
			}
		}
		// input should get builtins and runtime arrays
		if (var.storage == StorageClassInput ||
		    var.storage == StorageClassWorkgroup) // && is_builtin_variable(var) && want_builtins)
		{
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += to_expression(var_id);
		}
	}

	if (!ep_args.empty() && append_comma)
		ep_args += ", ";

	return ep_args;
}

string CompilerISPC::read_access_chain(const SPIRAccessChain &chain)
{
	auto &type = get<SPIRType>(chain.basetype);

	SPIRType target_type;
	target_type.basetype = SPIRType::UInt;
	target_type.vecsize = type.vecsize;
	target_type.columns = type.columns;

	// FIXME: Transposition?
	if (type.columns != 1)
		SPIRV_CROSS_THROW("Reading matrices from ByteAddressBuffer not yet supported.");

	if (type.basetype == SPIRType::Struct)
		return "blurgh structs from ByteAddressBuffer";
	//        SPIRV_CROSS_THROW("Reading structs from ByteAddressBuffer not yet supported.");

	if (type.width != 32)
		SPIRV_CROSS_THROW("Reading types other than 32-bit from ByteAddressBuffer not yet supported.");

	const char *load_op = nullptr;
	switch (type.vecsize)
	{
	case 1:
		load_op = "load1";
		break;
	case 2:
		load_op = "load2";
		break;
	case 3:
		load_op = "load3";
		break;
	case 4:
		load_op = "load4";
		break;
	default:
		SPIRV_CROSS_THROW("Unknown vector size.");
	}

	// currently have     varying float4 pos = uintBitsToFloat(oldPosVelo.Load4(DTid.x * 32 + 0));
	// want               varying float4 pos = float4(oldPosVelo._data[DTid.x].pos);

	//chain.base = oldPosVelo
	//chain.dynamic_index = DTid.x * 32;
	//chain.static_index = 0
	//bitcase = uintBitsToFloat

	auto load_expr = join(load_op, "(", chain.base, ")");

	return load_expr;
}

void CompilerISPC::emit_load(const Instruction &instruction)
{
	auto ops = stream(instruction);

	auto *chain = maybe_get<SPIRAccessChain>(ops[2]);
	if (chain)
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];

		auto load_expr = read_access_chain(*chain);

		bool forward = should_forward(ptr) && forced_temporaries.find(id) == end(forced_temporaries);
		auto &e = emit_op(result_type, id, load_expr, forward, true);
		e.need_transpose = false; // TODO: Forward this somehow.
		register_read(id, ptr, forward);
	}
	else
		CompilerGLSL::emit_instruction(instruction);
}

void CompilerISPC::emit_store(const Instruction &instruction)
{
	auto ops = stream(instruction);
	auto *chain = maybe_get<SPIRAccessChain>(ops[0]);
	auto &type = expression_type(ops[0]);
	//    if (chain && type.vecsize > 1)
	if (0)
	{

		SPIRType target_type;
		target_type.basetype = SPIRType::UInt;
		target_type.vecsize = type.vecsize;
		target_type.columns = type.columns;

		const char *store_op = nullptr;
		switch (type.vecsize)
		{
		case 1:
			store_op = "store";
			break;
		case 2:
			store_op = "store2";
			break;
		case 3:
			store_op = "store3";
			break;
		case 4:
			store_op = "store4";
			break;
		default:
			SPIRV_CROSS_THROW("Unknown vector size.");
		}

		if (type.columns != 1)
			SPIRV_CROSS_THROW("Writing matrices to RWByteAddressBuffer not yet supported.");
		if (type.basetype == SPIRType::Struct)
			SPIRV_CROSS_THROW("Writing structs to RWByteAddressBuffer not yet supported.");
		if (type.width != 32)
			SPIRV_CROSS_THROW("Writing types other than 32-bit to RWByteAddressBuffer not yet supported.");

		// currently have         newPosVelo._data[DTid.x].pos.store4(DTid.x * 32 + 0, floatBitsToUint(pos));
		// want               newPosVelo._data[DTid.x].pos = store4(pos);

		//chain.base = newPosVelo

		auto store_expr = to_expression(ops[1]);
		statement(chain->base, " = ", store_op, "(", store_expr, ");");

		register_write(ops[0]);
	}
	else
		CompilerGLSL::emit_instruction(instruction);
}

void CompilerISPC::emit_access_chain(const Instruction &instruction)
{
	auto ops = stream(instruction);
	uint32_t length = instruction.length;

	bool need_byte_access_chain = false;
	auto &type = expression_type(ops[2]);
	const SPIRAccessChain *chain = nullptr;
	if (has_decoration(type.self, DecorationBufferBlock))
	{
		// If we are starting to poke into an SSBO, we are dealing with ByteAddressBuffers, and we need
		// to emit SPIRAccessChain rather than a plain SPIRExpression.
		uint32_t chain_arguments = length - 3;
		if (chain_arguments > type.array.size())
			need_byte_access_chain = true;
	}
	else
	{
		// Keep tacking on an existing access chain.
		chain = maybe_get<SPIRAccessChain>(ops[2]);
		if (chain)
			need_byte_access_chain = true;
	}

	if (need_byte_access_chain)
	{
		auto *var = maybe_get<SPIRVariable>(ops[2]);
		if (var)
			flush_variable_declaration(var->self);

		// If the base is immutable, the access chain pointer must also be.
		// If an expression is mutable and forwardable, we speculate that it is immutable.
		bool need_transpose;
		auto ac = access_chain(ops[2], &ops[3], length - 3, get<SPIRType>(ops[0]), &need_transpose);

		uint32_t to_plain_buffer_length = type.array.size();

		string base = ac;

		auto *basetype = &type;

		// Start traversing type hierarchy at the proper non-pointer types.
		while (basetype->pointer)
		{
			assert(basetype->parent_type);
			basetype = &get<SPIRType>(basetype->parent_type);
		}

		// Traverse the type hierarchy down to the actual buffer types.
		for (uint32_t i = 0; i < to_plain_buffer_length; i++)
		{
			assert(basetype->parent_type);
			basetype = &get<SPIRType>(basetype->parent_type);
		}
		uint32_t matrix_stride = 0;
		auto offsets =
		    flattened_access_chain_offset(*basetype, &ops[3 + to_plain_buffer_length],
		                                  length - 3 - to_plain_buffer_length, 0, 1, &need_transpose, &matrix_stride);

		auto &e = set<SPIRAccessChain>(ops[1], ops[0], type.storage, base, offsets.first, offsets.second);
		if (chain)
		{
			e.dynamic_index += chain->dynamic_index;
			e.static_index += chain->static_index;
		}

		e.immutable = should_forward(ops[2]);
	}
	else
	{
		CompilerGLSL::emit_instruction(instruction);
	}
}

string CompilerISPC::layout_for_member(const SPIRType &, uint32_t)
{
	return "";
}

//
// Now for the codegen for the stdlib ISPC include file
//
void CompilerISPC::codegen_default_binary_op(std::string type, uint32_t width, std::string op)
{
	vector<string> varying = { "uniform ", "varying " };
	vector<string> arg_swizzles = { ".x", ".y", ".z", ".w" };

	for (auto &arg1 : varying)
	{
		for (auto &arg2 : varying)
		{
			auto &ret_varying = (arg1 != arg2) ? varying[1] : arg1;

			// scalar
			std::string s = join("static SPIRV_INLINE ", ret_varying, type, width, " operator", op, "(", arg1, type,
			                     width, " a, ", arg2, type, " b) { ");
			s += join(ret_varying, type, width, " ret = { ");
			for (uint32_t ii = 0; ii < width; ii++)
			{
				s += join("a", arg_swizzles[ii], " ", op, " b ");
				if ((ii + 1) < width)
					s += ", ";
			}
			s += join("}; return ret; }");
			statement(s);

			// vector
			s = join("static SPIRV_INLINE ", ret_varying, type, width, " operator", op, "(", arg1, type, width, " a, ",
			         arg2, type, width, " b) { ");
			s += join(ret_varying, type, width, " ret = { ");
			for (uint32_t ii = 0; ii < width; ii++)
			{
				s += join("a", arg_swizzles[ii], " ", op, " b", arg_swizzles[ii], " ");
				if ((ii + 1) < width)
					s += ", ";
			}
			s += join("}; return ret; }");
			statement(s);
		}
	}
	statement("");
}

void CompilerISPC::codegen_load_op(std::string type, uint32_t width)
{
	vector<string> varying = { "uniform ", "varying " };
	vector<string> arg_swizzles = { ".x", ".y", ".z", ".w" };

	for (auto &v : varying)
	{
		statement("static SPIRV_INLINE ", v, type, width, " load", width, "(", v, type, width, " a)");
		begin_scope();
		string args;
		for (uint32_t ii = 0; ii < width; ii++)
		{
			args += join("a", arg_swizzles[ii]);
			if ((ii + 1) < width)
				args += ", ";
		}
		statement("return ", type, width, "(", args, ");");
		end_scope();
		statement("");
	}
}

void CompilerISPC::codegen_store_op(std::string type, uint32_t width)
{
	vector<string> varying = { "uniform ", "varying " };
	vector<string> arg_swizzles = { ".x", ".y", ".z", ".w" };
	for (auto &v : varying)
	{
		statement("static SPIRV_INLINE ", v, type, width, " store", width, "(", v, type, width, " a)");
		begin_scope();
		statement(v, type, width, " ret;");

		string args;
		for (uint32_t ii = 0; ii < width; ii++)
		{
			statement("ret", arg_swizzles[ii], " = a", arg_swizzles[ii], ";");
		}
		statement("return ret;");
		end_scope();
		statement("");
	}
}

void CompilerISPC::codegen_default_structs(std::string type, uint32_t width)
{
	std::vector<string> arg_swizzles = { "x", "y", "z", "w" };
	statement("struct ", type, width);
	begin_scope();
	string args;
	for (uint32_t ii = 0; ii < width; ii++)
	{
		statement(type, " ", arg_swizzles[ii], ";");
	}
	end_scope_decl();
	statement("");
}

void CompilerISPC::codegen_default_pixel_structs(uint32_t width)
{
	std::vector<string> arg_swizzles = { "r", "g", "b", "a" };
	statement("struct pixel", width, "D");
	begin_scope();
	string args;
	for (uint32_t ii = 0; ii < width; ii++)
	{
		statement("unsigned int8 ", arg_swizzles[ii], ";");
	}
	end_scope_decl();
	statement("");
}

void CompilerISPC::codegen_default_image_structs(uint32_t width)
{
	statement("struct image", width, "D");
	begin_scope();
	statement("unsigned int width;");
	statement("unsigned int height;");
	statement("unsigned int stride;");
	statement("struct pixel4D* data;");
	end_scope_decl();
	statement("");
}

void CompilerISPC::codegen_constructor(std::string type, bool varying, uint32_t width, uint32_t arg_count,
                                       uint32_t arg_width[4])
{
	std::vector<string> arg_names = { "a", "b", "c", "d" };
	std::vector<string> arg_swizzles = { ".x", ".y", ".z", ".w" };

	std::vector<string> vector_names = { "varying ", "uniform " };
	std::string v = vector_names[varying ? 0 : 1];

	uint32_t function_count = 1;
	if (varying && arg_count > 1)
		function_count = 2;

	// For varying functions, create 2 variants
	// 1 - func (varying, varying)
	// 2 - func (varying, uniform)
	// Could do more, but that will suffice for now
	for (uint32_t ii = 0; ii < function_count; ii++)
	{
		// Create signature
		string args = "";
		for (uint32_t ac = 0; ac < arg_count; ac++)
		{
			if ((ac == arg_count - 1) && (function_count > 1))
				args += vector_names[ii];
			else
				args += v;

			args += join("const ", type); // varying const float
			if (arg_width[ac] > 1)
			{
				args += join(arg_width[ac]); // varying const float2
				args += "&"; // varying const float2&
			}
			args += join(" ", arg_names[ac]); // varying const float2& a
			if ((ac + 1) < arg_count)
				args += ", ";
		}

		// Create list initialiser
		string init;
		uint32_t required_initialisers = 0;
		while (required_initialisers < width)
		{
			for (uint32_t ac = 0; ac < arg_count; ac++)
			{
				for (uint32_t aw = 0; aw < arg_width[ac]; aw++)
				{
					required_initialisers++;

					init += arg_names[ac];
					if (arg_width[ac] > 1)
						init += arg_swizzles[aw];
					if (required_initialisers < width)
						init += ", ";
				}
			}
		}
		statement("static SPIRV_INLINE ", v, type, width, " ", type, width, "_init(", args, " ) { ", v, type, width,
		          " ret = { ", init, " }; return ret; }");
	}
}

#if 1
void CompilerISPC::codegen_matrix_constructor(std::string type, bool varying)
{
    std::vector<string> vector_names = { "varying ", "uniform " };
    std::string v = vector_names[varying ? 0 : 1];

    // variant 1 = pass in a single float - this is placed across the diagonal in an identity matrix
    statement("static SPIRV_INLINE ", v, "mat3 mat3_init(const ", v, " float a)");
    begin_scope();
    statement(v, " mat3 mat;");
    statement("mat.m[0] = float3_init(a, 0.0f, 0.0f);");
    statement("mat.m[1] = float3_init(0.0f, a, 0.0f);");
    statement("mat.m[2] = float3_init(0.0f, 0.0f, a);");
    statement("return mat;");
    end_scope();
    statement("");

    statement("static SPIRV_INLINE ", v, "mat4 mat4_init(const ", v, " float a)");
    begin_scope();
    statement(v, " mat4 mat;");
    statement("mat.m[0] = float4_init(a, 0.0f, 0.0f, 0.0f);");
    statement("mat.m[1] = float4_init(0.0f, a, 0.0f, 0.0f);");
    statement("mat.m[2] = float4_init(0.0f, 0.0f, a, 0.0f);");
    statement("mat.m[3] = float4_init(0.0f, 0.0f, 0.0f, a);");
    statement("return mat;");
    end_scope();
    statement("");

    // variant 2 = pass in vectors
    statement("static SPIRV_INLINE ", v, "mat3 mat3_init(const ", v, " float3& a, const ", v, " float3& b, const ", v, " float3& c)");
    begin_scope();
    statement(v, " mat3 mat;");
    statement("mat.m[0] = a;");
    statement("mat.m[1] = b;");
    statement("mat.m[2] = c;");
    statement("return mat;");
    end_scope();
    statement("");

    statement("static SPIRV_INLINE ", v, "mat4 mat4_init(const ", v, " float4& a, const ", v, " float4& b, const ", v, " float4& c, const ", v, " float4& d)");
    begin_scope();
    statement(v, " mat4 mat;");
    statement("mat.m[0] = a;");
    statement("mat.m[1] = b;");
    statement("mat.m[2] = c;");
    statement("mat.m[3] = d;");
    statement("return mat;");
    end_scope();
    statement("");

    // variant 3 = copy - done this way as ISPC won't allow arrays to be copied
    statement("static SPIRV_INLINE void mat3_copy(", v, " mat3& lhs, const ", v, " mat3& rhs)");
    begin_scope();
    statement("lhs.m[0] = rhs.m[0];");
    statement("lhs.m[1] = rhs.m[1];");
    statement("lhs.m[2] = rhs.m[2];");
    end_scope();
    statement("");

    statement("static SPIRV_INLINE void mat4_copy(", v, " mat4& lhs, const ", v, " mat4& rhs)");
    begin_scope();
    statement("lhs.m[0] = rhs.m[0];");
    statement("lhs.m[1] = rhs.m[1];");
    statement("lhs.m[2] = rhs.m[2];");
    statement("lhs.m[3] = rhs.m[3];");
    end_scope();
    statement("");
}
#endif

void CompilerISPC::codegen_cast_constructor(std::string src_type, std::string dst_type, bool varying, uint32_t width)
{
	std::vector<string> arg_swizzles = { ".x", ".y", ".z", ".w" };

	std::vector<string> vector_names = { "varying ", "uniform " };
	std::string v = vector_names[varying ? 0 : 1];

	// Create signature
	string args = "";
	args += v;

	args += join("const ", src_type); // varying const float
	if (width > 1)
	{
		args += join(width); // varying const float2
		args += "&"; // varying const float2&
	}
	args += join(" a"); // varying const float2& a

	// Create list initialiser
	string init;
	uint32_t required_initialisers = 0;
	while (required_initialisers < width)
	{
		for (uint32_t aw = 0; aw < width; aw++)
		{
			required_initialisers++;
			init += join(" (", v, dst_type, ")a");
			if (width > 1)
				init += arg_swizzles[aw];
			if (required_initialisers < width)
				init += ", ";
		}
	}
	statement("static SPIRV_INLINE ", v, dst_type, width, " ", dst_type, width, "_init(", args, " ) { ", v, dst_type,
	          width, " ret = { ", init, " }; return ret; }");
}

// varyings/vector widths are : return, arg1, arg2, arg3
void CompilerISPC::codegen_ternary_float_op_multiple_widths(
    std::string func_name, std::vector<std::vector<std::string>> &varyings,
    std::vector<std::vector<uint32_t>> vector_widths,
    const std::function<void(std::vector<std::string> varyings, std::vector<uint32_t> vector_widths)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_widths)
	{
		for (auto &v : varyings)
		{
			// pass by ref if non POD
			std::string op = join("static SPIRV_INLINE ", v[0], " float");
			if (w[0] > 1)
				op += join(w[0]);
			op += join(" ", func_name, "(", v[1], " float");
			if (w[1] > 1)
				op += join(w[1], "&");
			op += join(" a, ", v[2], " float");
			if (w[2] > 1)
				op += join(w[2], "&");
			op += join(" b, ", v[3], " float");
			if (w[3] > 1)
				op += join(w[3], "&");
			op += join(" c)");

			statement(op);
			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2, arg3
void CompilerISPC::codegen_ternary_float_op(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a, ", v[2],
				          " float b, ", v[3], " float c)");
			else
				statement("static SPIRV_INLINE ", v[0], " float", w, " ", func_name, "(", v[1], " float", w, "& a, ",
				          v[2], " float", w, "& b, ", v[3], " float", w, "& c)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_ternary_float_op_simple(std::string func_name,
                                                   std::vector<std::vector<std::string>> &varyings,
                                                   std::vector<uint32_t> vector_width)
{
	codegen_ternary_float_op(func_name, varyings, vector_width, [&](std::vector<std::string> v, int vw) {
		std::string s;
		if (vw == 1)
			s = join(v[0], " float ret = { ", func_name, "(a.x, b.x, c.x)");
		else
			s = join(v[0], " float", vw, " ret = { ", func_name, "(a.x, b.x, c.x)");
		if (vw > 1)
			s += join(", ", func_name, "(a.y, b.y, c.y)");
		if (vw > 2)
			s += join(", ", func_name, "(a.z, b.z, c.z)");
		if (vw > 3)
			s += join(", ", func_name, "(a.w, b.w, c.w)");
		s += " }; return ret;";
		statement(s);
	});
};

// varyings/vector widths are : return, arg1, arg2, arg3
void CompilerISPC::codegen_ternary_float_op_scalar_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a, ", v[2],
				          " float b, ", v[3], " float c)");
			else
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float", w, "& a, ", v[2],
				          " float", w, "& b, ", v[3], " float", w, "& c)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_binary_float_op(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a, ", v[2],
				          " float b)");
			else
				statement("static SPIRV_INLINE ", v[0], " float", w, " ", func_name, "(", v[1], " float", w, "& a, ",
				          v[2], " float", w, "& b)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_binary_float_op_simple(std::string func_name,
                                                  std::vector<std::vector<std::string>> &varyings,
                                                  std::vector<uint32_t> vector_width)
{
	codegen_binary_float_op(func_name, varyings, vector_width, [&](std::vector<std::string> v, int vw) {
		std::string s;
		if (vw == 1)
			s = join(v[0], " float ret = { ", func_name, "(a.x, b.x)");
		else
			s = join(v[0], " float", vw, " ret = { ", func_name, "(a.x, b.x)");
		if (vw > 1)
			s += join(", ", func_name, "(a.y, b.y)");
		if (vw > 2)
			s += join(", ", func_name, "(a.z, b.z)");
		if (vw > 3)
			s += join(", ", func_name, "(a.w, b.w)");
		s += " }; return ret;";
		statement(s);
	});
};

void CompilerISPC::codegen_binary_float_op_scalar_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a, ", v[2],
				          " float b)");
			else
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float", w, "& a, ", v[2],
				          " float", w, "& b)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_unary_float_op(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a)");
			else
				statement("static SPIRV_INLINE ", v[0], " float", w, " ", func_name, "(", v[1], " float", w, "& a)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_unary_float_op_scalar_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float a)");
			else
				statement("static SPIRV_INLINE ", v[0], " float ", func_name, "(", v[1], " float", w, "& a)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1
void CompilerISPC::codegen_unary_float_op_simple(std::string func_name, std::vector<std::vector<std::string>> &varyings,
                                                 std::vector<uint32_t> vector_width)
{
	codegen_unary_float_op(func_name, varyings, vector_width, [&](std::vector<std::string> v, int vw) {
		std::string s;
		if (vw == 1)
			s = join(v[0], " float ret = { ", func_name, "(a.x)");
		else
			s = join(v[0], " float", vw, " ret = { ", func_name, "(a.x)");
		if (vw > 1)
			s += join(", ", func_name, "(a.y)");
		if (vw > 2)
			s += join(", ", func_name, "(a.z)");
		if (vw > 3)
			s += join(", ", func_name, "(a.w)");
		s += " }; return ret;";
		statement(s);
	});
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_unary_float_op_scalar_bool_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " bool ", func_name, "(", v[1], " float a)");
			else
				statement("static SPIRV_INLINE ", v[0], " bool ", func_name, "(", v[1], " float", w, "& a)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1
void CompilerISPC::codegen_unary_float_op_int_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " int ", func_name, "(", v[1], " float a)");
			else
				statement("static SPIRV_INLINE ", v[0], " int", w, " ", func_name, "(", v[1], " float", w, "& a)");

			begin_scope();
			func(v, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_unary_op_scalar_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::string> &types,
    std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types, uint32_t vector_width)>
        &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " ", types[0], " ", func_name, "(", v[1], " ", types[1], " a)");
			else
				statement("static SPIRV_INLINE ", v[0], " ", types[0], " ", func_name, "(", v[1], " ", types[1], w,
				          "& a)");

			begin_scope();
			func(v, types, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_binary_op_scalar_return(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::string> &types,
    std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types, uint32_t vector_width)>
        &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " ", types[0], " ", func_name, "(", v[1], " ", types[1], " a, ",
				          v[2], " ", types[2], " b)");
			else
				statement("static SPIRV_INLINE ", v[0], " ", types[0], " ", func_name, "(", v[1], " ", types[1], w,
				          "& a, ", v[2], " ", types[2], w, "& b)");

			begin_scope();
			func(v, types, w);
			end_scope();
			statement("");
		}
	}
};

void CompilerISPC::codegen_binary_op(
    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::string> &types,
    std::vector<uint32_t> vector_width,
    const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types, uint32_t vector_width)>
        &func)
{
	statement("//////////////////////////////");
	statement("// ", func_name);
	statement("//////////////////////////////");

	for (auto &w : vector_width)
	{
		for (auto &v : varyings)
		{
			if (w == 1)
				statement("static SPIRV_INLINE ", v[0], " ", types[0], " ", func_name, "(", v[1], " ", types[1], " a, ",
				          v[2], " ", types[2], " b)");
			else
				statement("static SPIRV_INLINE ", v[0], " ", types[0], w, " ", func_name, "(", v[1], " ", types[1], w,
				          "& a, ", v[2], " ", types[2], w, "& b)");

			begin_scope();
			func(v, types, w);
			end_scope();
			statement("");
		}
	}
};

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_matrix_multiply(uint32_t dim, std::vector<std::vector<std::string>> &varyings)
{
    auto mat_mul_col = [&](const uint32_t dim, const uint32_t col)
    {
        std::vector<string> rows = { "x", "y", "z", "w" };

        auto str = join("ret.m[" + convert_to_string(col) + "] = "
            "lhs.m[0] * rhs.m[" + convert_to_string(col) + "]." + rows[0] +
            " + lhs.m[1] * rhs.m[" + convert_to_string(col) + "]." + rows[1]);
        if (dim > 2)
            str += join(" + lhs.m[2] * rhs.m[" + convert_to_string(col) + "]." + rows[2]);
        if (dim > 3)
            str += join(" + lhs.m[3] * rhs.m[" + convert_to_string(col) + "]." + rows[3]);
        str += ";";
        statement(str);
    };

    statement("");
    statement("//////////////////////////////");
    auto str = join(convert_to_string(dim) + "x" + convert_to_string(dim));
    statement("// Matrix Multiply " + str);
    statement("//////////////////////////////");
    statement("");

    auto matN = join("mat" + convert_to_string(dim));

    for (auto &v : varyings)
    {
        statement(
            "static SPIRV_INLINE " + v[0] + " " + matN + " operator*(" + v[1] + " " + matN + " &lhs, " + v[2] + " " + matN + " &rhs)");
        begin_scope();
        statement(v[0] + " " + matN + " ret;");
        for (uint32_t col = 0; col < dim; col++)
        {
            mat_mul_col(dim, col);
        }
        statement("return ret;");
        end_scope();
        statement("");
    }
}

// Inverse
void CompilerISPC::codegen_matrix_inverse(std::string &varying)
{
    statement("");
    statement("//////////////////////////////");
    statement("// Matrix Inverse ");
    statement("//////////////////////////////");
    statement("");

    statement("// Returns the determinant of a 2x2 matrix.");
    statement("static SPIRV_INLINE " + varying + " float determinantMat2(" + varying + " float a1, " + varying + " float a2, " + varying + " float b1, " + varying + " float b2)");
    begin_scope();
    statement("return a1 * b2 - b1 * a2;");
    end_scope();
    statement("");

    statement("// Returns the determinant of a 3x3 matrix.");
    statement("static SPIRV_INLINE " + varying + " float determinantMat3(" + varying + " float a1, " + varying + " float a2, " + varying + " float a3, " + varying + " float b1, " + varying + " float b2, " + varying + " float b3, " + varying + " float c1, " + varying + " float c2, " + varying + " float c3)");
    begin_scope();
    statement("return a1 * determinantMat2(b2, b3, c2, c3) - b1 * determinantMat2(a2, a3, c2, c3) + c1 * determinantMat2(a2, a3, "
        "b2, b3);");
    end_scope();
    statement("");
    statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
    statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
    statement("static SPIRV_INLINE " + varying + " mat4 inverse(" + varying + " mat4 m)");
    begin_scope();
    statement(varying + " mat4 adj;	// The adjoint matrix (inverse after dividing by determinant)");
    statement_no_indent("");
    statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
    statement("adj.m[0].x =  determinantMat3(m.m[1].y, m.m[1].z, m.m[1].w, m.m[2].y, m.m[2].z, m.m[2].w, m.m[3].y, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[0].y = -determinantMat3(m.m[0].y, m.m[0].z, m.m[0].w, m.m[2].y, m.m[2].z, m.m[2].w, m.m[3].y, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[0].z =  determinantMat3(m.m[0].y, m.m[0].z, m.m[0].w, m.m[1].y, m.m[1].z, m.m[1].w, m.m[3].y, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[0].w = -determinantMat3(m.m[0].y, m.m[0].z, m.m[0].w, m.m[1].y, m.m[1].z, m.m[1].w, m.m[2].y, m.m[2].z, "
        "m.m[2].w);");
    statement_no_indent("");
    statement("adj.m[1].x = -determinantMat3(m.m[1].x, m.m[1].z, m.m[1].w, m.m[2].x, m.m[2].z, m.m[2].w, m.m[3].x, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[1].y =  determinantMat3(m.m[0].x, m.m[0].z, m.m[0].w, m.m[2].x, m.m[2].z, m.m[2].w, m.m[3].x, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[1].z = -determinantMat3(m.m[0].x, m.m[0].z, m.m[0].w, m.m[1].x, m.m[1].z, m.m[1].w, m.m[3].x, m.m[3].z, "
        "m.m[3].w);");
    statement("adj.m[1].w =  determinantMat3(m.m[0].x, m.m[0].z, m.m[0].w, m.m[1].x, m.m[1].z, m.m[1].w, m.m[2].x, m.m[2].z, "
        "m.m[2].w);");
    statement_no_indent("");
    statement("adj.m[2].x =  determinantMat3(m.m[1].x, m.m[1].y, m.m[1].w, m.m[2].x, m.m[2].y, m.m[2].w, m.m[3].x, m.m[3].y, "
        "m.m[3].w);");
    statement("adj.m[2].y = -determinantMat3(m.m[0].x, m.m[0].y, m.m[0].w, m.m[2].x, m.m[2].y, m.m[2].w, m.m[3].x, m.m[3].y, "
        "m.m[3].w);");
    statement("adj.m[2].z =  determinantMat3(m.m[0].x, m.m[0].y, m.m[0].w, m.m[1].x, m.m[1].y, m.m[1].w, m.m[3].x, m.m[3].y, "
        "m.m[3].w);");
    statement("adj.m[2].w = -determinantMat3(m.m[0].x, m.m[0].y, m.m[0].w, m.m[1].x, m.m[1].y, m.m[1].w, m.m[2].x, m.m[2].y, "
        "m.m[2].w);");
    statement_no_indent("");
    statement("adj.m[3].x = -determinantMat3(m.m[1].x, m.m[1].y, m.m[1].z, m.m[2].x, m.m[2].y, m.m[2].z, m.m[3].x, m.m[3].y, "
        "m.m[3].z);");
    statement("adj.m[3].y =  determinantMat3(m.m[0].x, m.m[0].y, m.m[0].z, m.m[2].x, m.m[2].y, m.m[2].z, m.m[3].x, m.m[3].y, "
        "m.m[3].z);");
    statement("adj.m[3].z = -determinantMat3(m.m[0].x, m.m[0].y, m.m[0].z, m.m[1].x, m.m[1].y, m.m[1].z, m.m[3].x, m.m[3].y, "
        "m.m[3].z);");
    statement("adj.m[3].w =  determinantMat3(m.m[0].x, m.m[0].y, m.m[0].z, m.m[1].x, m.m[1].y, m.m[1].z, m.m[2].x, m.m[2].y, "
        "m.m[2].z);");
    statement_no_indent("");
    statement("// Calculate the determinant as a combination of the cofactors of the first row.");
    statement(varying + " float det = (adj.m[0].x * m.m[0].x) + (adj.m[0].y * m.m[1].x) + (adj.m[0].z * m.m[2].x) + (adj.m[0].w "
        "* m.m[3].x);");
    statement_no_indent("");
    statement("// Divide the classical adjoint matrix by the determinant.");
    statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
    statement("return (det != 0.0f) ? adj * (1.0f / det) : m;");
    end_scope();
    statement("");

    statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
    statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
    statement("static SPIRV_INLINE " + varying + " mat3 inverse(" + varying + " mat3 m)");
    begin_scope();
    statement(varying + " mat3 adj;	// The adjoint matrix (inverse after dividing by determinant)");
    statement_no_indent("");
    statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
    statement("adj.m[0].x =  determinantMat2(m.m[1].y, m.m[1].z, m.m[2].y, m.m[2].z);");
    statement("adj.m[0].y = -determinantMat2(m.m[0].y, m.m[0].z, m.m[2].y, m.m[2].z);");
    statement("adj.m[0].z =  determinantMat2(m.m[0].y, m.m[0].z, m.m[1].y, m.m[1].z);");
    statement_no_indent("");
    statement("adj.m[1].x = -determinantMat2(m.m[1].x, m.m[1].z, m.m[2].x, m.m[2].z);");
    statement("adj.m[1].y =  determinantMat2(m.m[0].x, m.m[0].z, m.m[2].x, m.m[2].z);");
    statement("adj.m[1].z = -determinantMat2(m.m[0].x, m.m[0].z, m.m[1].x, m.m[1].z);");
    statement_no_indent("");
    statement("adj.m[2].x =  determinantMat2(m.m[1].x, m.m[1].y, m.m[2].x, m.m[2].y);");
    statement("adj.m[2].y = -determinantMat2(m.m[0].x, m.m[0].y, m.m[2].x, m.m[2].y);");
    statement("adj.m[2].z =  determinantMat2(m.m[0].x, m.m[0].y, m.m[1].x, m.m[1].y);");
    statement_no_indent("");
    statement("// Calculate the determinant as a combination of the cofactors of the first row.");
    statement(varying + " float det = (adj.m[0].x * m.m[0].x) + (adj.m[0].y * m.m[1].x) + (adj.m[0].z * m.m[2].x);");
    statement_no_indent("");
    statement("// Divide the classical adjoint matrix by the determinant.");
    statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
    statement("return (det != 0.0f) ? adj * (1.0f / det) : m;");
    end_scope();
    statement("");

}


// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_matrix_transpose(uint32_t dim, std::vector<std::vector<std::string>> &varyings)
{
    auto mat_trans_col_row = [&](const uint32_t dim, const uint32_t col, const uint32_t row)
    {
        std::vector<string> rows = { "x", "y", "z", "w" };
        auto str = join("ret.m[" + convert_to_string(col) + "] = float" + convert_to_string(dim) + "(rhs.m[0]." + rows[row] + ", rhs.m[1]." + rows[row]);
        if (dim > 2)
            str += join(", rhs.m[2]." + rows[row]);
        if (dim > 3)
            str += join(", rhs.m[3]." + rows[row]);
        str += ");";
        statement(str);
    };

    statement("");
    statement("//////////////////////////////");
    auto str = join(convert_to_string(dim) + "x" + convert_to_string(dim));
    statement("// Matrix Transpose " + str);
    statement("//////////////////////////////");
    statement("");

    auto matN = join("mat" + convert_to_string(dim));

    for (auto &v : varyings)
    {
        statement(
            "static SPIRV_INLINE " + v[0] + " " + matN + " transpose(" + v[1] + " " + matN + " &rhs)");
        begin_scope();
        statement(v[0] + " " + matN + " ret;");
        for (uint32_t col = 0; col < dim; col++)
        {
            mat_trans_col_row(dim, col, col);
        }
        statement("return ret;");
        end_scope();
        statement("");
    }
}

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_matrix_times_scalar(uint32_t dim, std::vector<std::vector<std::string>> &varyings)
{
    auto mat_times_scalar_col = [&](const uint32_t dim, const uint32_t col, const uint32_t row)
    {
        std::vector<string> rows = { "x", "y", "z", "w" };
        auto str = join("ret.m[" + convert_to_string(col) + "] = lhs.m[" + convert_to_string(col) + "] * rhs;");
        statement(str);
    };
    auto scalar_times_mat_col = [&](const uint32_t dim, const uint32_t col, const uint32_t row)
    {
        std::vector<string> rows = { "x", "y", "z", "w" };
        auto str = join("ret.m[" + convert_to_string(col) + "] = rhs.m[" + convert_to_string(col) + "] * lhs;");
        statement(str);
    };

    statement("");
    statement("//////////////////////////////");
    auto str = join(convert_to_string(dim) + "x" + convert_to_string(dim));
    statement("// Matrix Times Scalar " + str);
    statement("//////////////////////////////");
    statement("");

    auto matN = join("mat" + convert_to_string(dim));

    for (auto &v : varyings)
    {
        statement(
            "static SPIRV_INLINE " + v[0] + " " + matN + " operator*(" + v[1] + " " + matN + " &lhs, " + v[2] + " float &rhs)");
        begin_scope();
        statement(v[0] + " " + matN + " ret;");
        for (uint32_t col = 0; col < dim; col++)
        {
            mat_times_scalar_col(dim, col, col);
        }
        statement("return ret;");
        end_scope();
        statement("");

        statement(
            "static SPIRV_INLINE " + v[0] + " " + matN + " operator*(" + v[1] + " float &lhs, " + v[2] + " " + matN + " &rhs)");
        begin_scope();
        statement(v[0] + " " + matN + " ret;");
        for (uint32_t col = 0; col < dim; col++)
        {
            scalar_times_mat_col(dim, col, col);
        }
        statement("return ret;");
        end_scope();
        statement("");
    }
}

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_vector_times_matrix(uint32_t dim, std::vector<std::vector<std::string>> &varyings)
{
    // Assume column major matrices and row vectors
    //
    //               | a, d, g |
    // | x, y, z | x | b, e, h | = | dot(mat[0], vec), dot(mat[1], vec), dot(mat[2], vec) | 
    //               | c, f, i |
    //
    //
    statement("");
    statement("////////////////////////////////////////");
    auto str = join(convert_to_string(dim) + "x" + convert_to_string(dim));
    statement("// Row Vector Times Column Matrix " + str);
    statement("////////////////////////////////////////");
    statement("");

    auto floatN = join("float" + convert_to_string(dim));
    auto matN = join("mat" + convert_to_string(dim));

    for (auto &v : varyings)
    {
        statement(
            "static SPIRV_INLINE " + v[0] + " " + floatN + " operator*(" + v[1] + " " + floatN + " &lhs, " + v[2] + " " + matN + " &rhs)");
        begin_scope();
        statement(v[0] + " " + floatN + " ret;");

        string str = "ret = " + floatN + "(";
        for (uint32_t c = 0; c < dim; c++)
        {
            str += " dot(lhs, rhs.m[" + convert_to_string(c) + "])";
            if (c < (dim - 1))
                str += ",";
        }
        str += ");";
        statement(str);
        statement("return ret;");
        end_scope();
        statement("");
    }
}

// varyings/vector widths are : return, arg1, arg2
void CompilerISPC::codegen_matrix_times_vector(uint32_t dim, std::vector<std::vector<std::string>> &varyings)
{
    // Assume column major matrices and column vectors
    //
    // | a, d, g |   | x |   | dot(float3(a, d, g), vec) |
    // | b, e, h | x | y | = | dot(float3(b, e, h), vec) |
    // | c, f, i |   | z |   | dot(float3(c, f, i), vec) |
    //
    //
    auto mat_times_vec_row = [&](const uint32_t dim, const uint32_t row)
    {
        std::vector<string> rows = { "x", "y", "z", "w" };

        string dotp = "dot(float" + convert_to_string(dim) + "(";
        for (uint32_t c = 0; c < dim; c++)
        {
            dotp += " lhs.m[" + convert_to_string(c) + "]." + rows[row];
            if (c < (dim - 1))
                dotp += ",";
        }
        dotp += " ), rhs );";

        auto str = join("ret." + rows[row] + " = " + dotp);
        statement(str);
    };

    statement("");
    statement("////////////////////////////////////////////");
    auto str = join(convert_to_string(dim) + "x" + convert_to_string(dim));
    statement("// Column Matrix Times Column Vector " + str);
    statement("////////////////////////////////////////////");
    statement("");

    auto floatN = join("float" + convert_to_string(dim));
    auto matN = join("mat" + convert_to_string(dim));

    for (auto &v : varyings)
    {
        statement(
            "static SPIRV_INLINE " + v[0] + " " + floatN + " operator*(" + v[1] + " " + matN + " &lhs, " + v[2] + " " + floatN + " &rhs)");
        begin_scope();
        statement(v[0] + " " + floatN + " ret;");
        for (uint32_t row = 0; row < dim; row++)
        {
            mat_times_vec_row(dim, row);
        }
        statement("return ret;");
        end_scope();
        statement("");
    }
}

void CompilerISPC::emit_stdlib()
{
    statement("//////////////////////////////");
    statement("// This ISPC spirv stdlib kernel is autogenerated by spirv-cross.");
    statement("//////////////////////////////");

    if (debug)
        statement("#define SPIRV_INLINE");
    else
        statement("#define SPIRV_INLINE inline");

    statement("");
    statement("//////////////////////////////");
    statement("// Default Types");
    statement("//////////////////////////////");
    for (const string &t : std::vector<string>{ "float", "int", "bool" })
    {
        for (const uint32_t &w : std::vector<uint32_t>{ 1, 2, 3, 4 })
        {
            codegen_default_structs(t, w);
        }
    }
    statement("struct mat3 { float3 m[3]; };");
    statement("struct mat4 { float4 m[4]; };");

    statement("");

    statement("//////////////////////////////");
    statement("// Default Image Types");
    statement("//////////////////////////////");
    codegen_default_pixel_structs(4);
    codegen_default_image_structs(2);

    // These provide us with some simple codegen to cast uniforms to varyings.
    // Is a no-op when casting a varying to a varying.
    statement("//////////////////////////////");
    statement("// Default Varying Casts");
    statement("//////////////////////////////");
    for (const string &t : std::vector<string>{ "float", "int" }) //"bool" There seems to be an issue with bools
    {
        for (const string &v : std::vector<string>{ "uniform", "varying" })
        {
            for (const uint32_t &w : std::vector<uint32_t>{ 1, 2, 3, 4 })
            {
                std::string op = join("static SPIRV_INLINE varying ", t);
                if (w > 1)
                    op += join(w);
                op += join(" to_varying(");
                op += join(v, " ", t);
                if (w > 1)
                    op += join(w, "&");
                op += join(" a) { return (varying ", t);
                if (w > 1)
                    op += join(w);
                op += ")a; }";
                statement(op);
            }
        }
    }
    statement("");

    statement("//////////////////////////////");
    statement("// Default Simple Casts That Honour Varying/Uniforms");
    statement("//////////////////////////////");
    for (const string &t : std::vector<string>{ "float", "int" }) //"bool" There seems to be an issue with bools
    {
        for (const string &v : std::vector<string>{ "uniform", "varying" })
        {
            for (const string &t2 : std::vector<string>{ "float", "int", "bool" })
            {
                std::string op1 = join("static SPIRV_INLINE const ", v, " ", t, " ", t, "_cast(const ", v, " ", t2,
                    "& a) { return (const ", v, " ", t, ") a; }");
                std::string op2 = join("static SPIRV_INLINE ", v, " ", t, " ", t, "_cast(", v, " ", t2,
                    "& a) { return (", v, " ", t, ") a; }");
                statement(op1);
                statement(op2);
            }
        }
    }
    statement("");

    statement("//////////////////////////////");
    statement("// Default Constructors");
    statement("//////////////////////////////");
    for (const string &t : std::vector<string>{ "float", "int", "bool" })
    {
        for (const bool &v : std::vector<bool>{ true, false })
        {
            for (const uint32_t &w : std::vector<uint32_t>{ 2, 3, 4 })
            {
                uint32_t arg_widths[][4] = {
                    { w, 1, 1, 1 }, // all
                    { 1, 1, 1, 1 }, // all
                    { 2, 1, 1, 1 }, // float 3 and larger
                    { 3, 1, 1, 1 }, // float 4 and larger
                };

                codegen_constructor(t, v, w, 1, arg_widths[0]);
                codegen_constructor(t, v, w, 1, arg_widths[1]);
                codegen_constructor(t, v, w, w, arg_widths[1]);

                if (w > 2)
                {
                    codegen_constructor(t, v, w, w - 1, arg_widths[2]);
                }
                if (w > 3)
                {
                    codegen_constructor(t, v, w, w - 2, arg_widths[3]);
                }

                // Allow for : float2(int2 a)
                if (t == "float")
                    codegen_cast_constructor("int", t, v, w);
                else if (t == "int")
                    codegen_cast_constructor("float", t, v, w);
            }
            statement("");
        }

        for (const uint32_t &w : std::vector<uint32_t>{ 2, 3, 4 })
            statement("#define ", t, w, "(...) ", t, w, "_init(__VA_ARGS__)");
        statement("");
    }
#if 1
    for (const string &t : std::vector<string>{ "float" })
    {
        for (const bool &v : std::vector<bool>{ true, false })
        {
            codegen_matrix_constructor(t, v);
        }
    }

    statement("");
    statement("#define mat3(...) mat3_init(__VA_ARGS__)");
    statement("#define mat4(...) mat4_init(__VA_ARGS__)");
    statement("");

#endif
    statement("");
    statement("//////////////////////////////");
    statement("// Default Operators");
    statement("//////////////////////////////");

    //    { "*", "/", "%", "+", "-" };
    for (const string &t : std::vector<string>{ "float", "int" })
    {
        for (const uint32_t &w : std::vector<uint32_t>{ 2, 3, 4 })
        {
            //            codegen_load_op(t, w);
            //            codegen_store_op(t, w);
            for (const string &bop : std::vector<string>{ "*", "/", "+", "-" })
            {
                codegen_default_binary_op(t, w, bop);
            }
        }
    }

    //
    // Ternary Op
    //
    {
        vector<vector<string>> default_varying = {
            { "varying", "varying", "varying", "varying" },
            { "uniform", "uniform", "uniform", "uniform" },
        };
        vector<vector<string>> complex_varying = {
            { "varying", "varying", "varying", "varying" }, { "varying", "uniform", "varying", "varying" },
            { "varying", "varying", "uniform", "varying" }, { "varying", "varying", "varying", "uniform" },
            { "uniform", "uniform", "uniform", "uniform" },
        };
        vector<vector<string>> mixed_varying = {
            { "varying", "varying", "uniform", "uniform" },
            { "uniform", "uniform", "uniform", "uniform" },
        };

        // NOTE the uniform args for clamp
        codegen_ternary_float_op_simple("clamp", mixed_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_ternary_float_op("mix", complex_varying, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return a + c * (b - a);"); });
        codegen_ternary_float_op("fma", default_varying, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return a * b + c;"); });
        codegen_ternary_float_op("smoothstep", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string> varyings, int) {
            statement(varyings[0], " float t = clamp((c - a) / (b - a), 0.0f, 1.0f);");
            statement("return t * t * (3.0f - 2.0f * t);");
        });
        codegen_ternary_float_op_simple("smoothstep", default_varying, vector<uint32_t>{ 2, 3, 4 });
#if 0
        // Refract has different argument widths, as the last argument is scalar
        // This is causing an issue in ISPC - not investigated yet.
        vector<vector<uint32_t>> widths = { { 2, 2, 2, 1 },{ 3, 3, 3, 1 },{ 4, 4, 4, 1 } };
        codegen_ternary_float_op_multiple_widths("refract", default_varying, widths, [&](std::vector<std::string> varyings, std::vector<uint32_t> vector_widths)
        {
            statement(varyings[0], " float", vector_widths[0], " ret;");
            statement(varyings[0], " float k = 1.0f - c * c * (1.0f - dot(b, a) * dot(b, a));");
            statement("if (k < 0.0f) ret = float", vector_widths[0], "_init(0.0f);");
            statement("else ret = float", vector_widths[0], "_init(c) * a - float", vector_widths[0], "_init(c * dot(b, a) + sqrt(k)) * b;");
            statement("return ret;");
        });
#endif
    }

    //
    // Binary Op
    //
    {
        vector<vector<string>> default_varying = {
            { "varying", "varying", "varying" },
            { "varying", "varying", "uniform" },
            { "varying", "uniform", "varying" },
            { "uniform", "uniform", "uniform" },
        };

        vector<vector<string>> mixed_varying = {
            { "varying", "varying", "uniform" },
            { "varying", "uniform", "varying" },
        };

        codegen_binary_float_op_scalar_return(
            "dot", default_varying, vector<uint32_t>{ 2, 3, 4 }, [&](std::vector<std::string>, int vector_width) {
            switch (vector_width)
            {
            case 4:
                statement("return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;");
                break;
            case 3:
                statement("return a.x * b.x + a.y * b.y + a.z * b.z;");
                break;
            case 2:
                statement("return a.x * b.x + a.y * b.y;");
                break;
            }
        });

        codegen_binary_float_op(
            "reflect", default_varying, vector<uint32_t>{ 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return a - b * (dot(a, b) * 2.0f);"); });

        // all varying or all uniform versions already exist
        codegen_binary_float_op("min", mixed_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string> varyings, int) {
            // uniforms need promoting to varying for the select to work and any implicit conversions
            if (varyings[0] == "varying")
            {
                statement("return min(to_varying(a), to_varying(b));");
            }
            else
            {
                statement("return min(a, b)");
            }
        });
        codegen_binary_float_op_simple("min", default_varying, vector<uint32_t>{ 2, 3, 4 });

        codegen_binary_float_op("max", mixed_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string> varyings, int) {
            // uniforms need promoting to varying for the select to work and any implicit conversions
            if (varyings[0] == "varying")
            {
                statement("return max(to_varying(a), to_varying(b));");
            }
            else
            {
                statement("return max(a, b)");
            }
        });
        codegen_binary_float_op_simple("max", default_varying, vector<uint32_t>{ 2, 3, 4 });

        codegen_binary_float_op("step", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string> varyings, int) {
            // uniforms need promoting to varying for the select to work and any implicit conversions
            if (varyings[0] == "varying")
            {
                statement("return to_varying(b) >= to_varying(a) ? 1.0f : 0.0f;");
            }
            else
            {
                statement("return b >= a ? 1.0f : 0.0f;");
            }
        });
        codegen_binary_float_op_simple("step", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_binary_float_op_simple("pow", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_binary_float_op("mod", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string> varyings, int) {
            // uniforms need promoting to varying for the select to work and any implicit conversions
            if (varyings[0] == "varying")
            {
                statement("varying float va = a;");
                statement("varying float vb = b;");
                statement("return va - vb * floor(va / vb);");
            }
            else
            {
                statement("return a - b * floor(a / b);");
            }
        });
        codegen_binary_float_op_simple("mod", default_varying, vector<uint32_t>{ 2, 3, 4 });

        // Return a bool
        vector<string> args = { "bool", "float", "float" };
        codegen_binary_op("notEqual", default_varying, args, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string> varyings, std::vector<std::string> types, int vector_width) {
            switch (vector_width)
            {
            case 1:
                statement("return a != b;");
                break;
            case 2:
                statement(varyings[0], " ", types[0], vector_width,
                    " ret = { a.x != b.x, a.y != b.y }; return ret;");
                break;
            case 3:
                statement(varyings[0], " ", types[0], vector_width,
                    " ret = { a.x != b.x, a.y != b.y, a.z != b.z }; return ret;");
                break;
            case 4:
                statement(varyings[0], " ", types[0], vector_width,
                    " ret = { a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w }; return ret;");
                break;
            }
        });

        // float3 version only
        codegen_binary_float_op("cross", default_varying, vector<uint32_t>{ 3 },
            [&](std::vector<std::string> varyings, int vector_width) {
            statement(varyings[0], " float", vector_width, " ret;");
            statement("ret.x = (a.y * b.z) - (a.z * b.y);");
            statement("ret.y = (a.z * b.x) - (a.x * b.z);");
            statement("ret.z = (a.x * b.y) - (a.y * b.x);");
            statement("return ret;");
        });
    }

    //
    // Unnary Op
    //
    {
        vector<vector<string>> default_varying = {
            { "varying", "varying" },
            { "uniform", "uniform" },
        };

        vector<vector<string>> varying_varying = { { "varying", "varying" } };

        codegen_unary_float_op_scalar_return(
            "length", default_varying, vector<uint32_t>{ 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return sqrt(dot(a, a));"); });

        codegen_unary_float_op_simple("abs", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("acos", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("asin", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("atan", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("cos", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("sin", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("tan", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("floor", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("round", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("ceil", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("log", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("log2", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return log(a) / log(2.0f);"); });
        codegen_unary_float_op_simple("log2", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("rcp", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("sqrt", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("rsqrt", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op_simple("exp", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("cosh", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return (exp(a) + exp(-a)) / 2.0f;"); });
        codegen_unary_float_op_simple("cosh", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("sinh", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return (exp(a) - exp(-a)) / 2.0f;"); });
        codegen_unary_float_op_simple("sinh", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("tanh", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return sinh(a) / cosh(a);"); });
        codegen_unary_float_op_simple("tanh", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("degrees", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return (180.0f * a) / PI;"); });
        codegen_unary_float_op_simple("degrees", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("radians", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return (PI * a) / 180.0f;"); });
        codegen_unary_float_op_simple("radians", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("fract", default_varying, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return a - floor(a);"); });
        codegen_unary_float_op("normalize", default_varying, vector<uint32_t>{ 2, 3, 4 },
            [&](std::vector<std::string>, int) { statement("return a / length(a);"); });
        codegen_unary_float_op("sign", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return (a < 0.0f) ? -1 : 1;"); });
        codegen_unary_float_op_simple("sign", default_varying, vector<uint32_t>{ 2, 3, 4 });
        codegen_unary_float_op("exp2", default_varying, vector<uint32_t>{ 1 },
            [&](std::vector<std::string>, int) { statement("return pow(2.0f, a);"); });
        codegen_unary_float_op_simple("exp2", default_varying, vector<uint32_t>{ 2, 3, 4 });

        // Int return type
        codegen_unary_float_op_int_return(
            "trunc", varying_varying, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string> varyings, int vector_width) {
            switch (vector_width)
            {
            case 1:
                statement("return (int)floor(a);");
                break;
            case 2:
                statement(varyings[0], " int", vector_width, " ret = { trunc(a.x), trunc(a.y) }; return ret;");
                break;
            case 3:
                statement(varyings[0], " int", vector_width,
                    " ret = { trunc(a.x), trunc(a.y), trunc(a.z) }; return ret;");
                break;
            case 4:
                statement(varyings[0], " int", vector_width,
                    " ret = { trunc(a.x), trunc(a.y), trunc(a.z), trunc(a.w)}; return ret;");
                break;
            }
        });

        // Scalar return
        vector<string> bool_bool = { "bool", "bool" };
        codegen_unary_op_scalar_return("all", varying_varying, bool_bool, vector<uint32_t>{ 2, 3, 4 },
            [&](std::vector<std::string>, std::vector<std::string>, int vector_width) {
            switch (vector_width)
            {
            case 2:
                statement("return all(a.x) && all(a.y);");
                break;
            case 3:
                statement("return all(a.x) && all(a.y) && all(a.z);");
                break;
            case 4:
                statement("return all(a.x) && all(a.y) && all(a.z) && all(a.w);");
                break;
            }
        });

        codegen_unary_op_scalar_return("any", varying_varying, bool_bool, vector<uint32_t>{ 2, 3, 4 },
            [&](std::vector<std::string>, std::vector<std::string>, int vector_width) {
            switch (vector_width)
            {
            case 2:
                statement("return any(a.x) || any(a.y);");
                break;
            case 3:
                statement("return any(a.x) || any(a.y) || any(a.z);");
                break;
            case 4:
                statement("return any(a.x) || any(a.y) || any(a.z) || any(a.w);");
                break;
            }
        });
    }

    //
    // Binary Op
    //
    {
        vector<vector<string>> default_varying = {
            { "varying", "varying", "varying" },
            { "varying", "varying", "uniform" },
            { "varying", "uniform", "varying" },
            { "uniform", "uniform", "uniform" },
        };

        // distance depends on length, so must be after it
        codegen_binary_float_op_scalar_return("distance", default_varying, vector<uint32_t>{ 1, 2, 3, 4 },
            [&](std::vector<std::string>, int vector_width) {
            switch (vector_width)
            {
            case 1:
                statement("return abs(a - b);");
                break;
            default:
                statement("return length(a - b);");
                break;
            }
        });
    }

    // Atomics
    // Currently implemented assumed atomic buffer counters. Probably needs work for non buffer based atomics
    {
        statement("");
        statement("//////////////////////////////");
        statement("// Atomics");
        statement("//////////////////////////////");
        vector<string> op = { "atomic_add", "atomic_subtract", "atomic_min", "atomic_max",
                              "atomic_and", "atomic_or",       "atomic_xor", "atomic_swap" };
        for (auto &o : op)
        {
            statement("static SPIRV_INLINE varying int ", o, "(uniform int * uniform ptr, varying int value)");
            begin_scope();
            statement("uniform int ret[programCount];");
            statement("foreach_active(instance)");
            begin_scope();
            statement("uniform int val = extract(value, instance);");
            statement("ret[instance] = ", o, "_global(ptr, val);");
            end_scope();
            statement("varying int vRet = *((varying int * uniform) &ret);");
            statement("return vRet;");
            end_scope();
            statement("");

            statement("static SPIRV_INLINE varying int ", o, "(uniform int * uniform ptr, uniform int value)");
            begin_scope();
            statement("uniform int ret[programCount];");
            statement("foreach_active(instance)");
            begin_scope();
            statement("ret[instance] = ", o, "_global(ptr, value);");
            end_scope();
            statement("varying int vRet = *((varying int * uniform) &ret);");
            statement("return vRet;");
            end_scope();
            statement("");

            statement("static SPIRV_INLINE varying int ", o, "(uniform int * varying vptr, varying int value)");
            begin_scope();
            statement("uniform int ret[programCount];");
            statement("uniform int ** uniform ptr = ((uniform int ** uniform) &vptr);");

            statement("foreach_active(instance)");
            begin_scope();
            statement("uniform int val = extract(value, instance);");
            statement("ret[instance] = ", o, "_global(ptr[instance], val);");
            end_scope();
            statement("varying int vRet = *((varying int * uniform) &ret);");
            statement("return vRet;");
            end_scope();
            statement("");

            statement("static SPIRV_INLINE varying int ", o, "(uniform int * varying vptr, uniform int value)");
            begin_scope();
            statement("uniform int ret[programCount];");
            statement("uniform int ** uniform ptr = ((uniform int ** uniform) &vptr);");
            statement("foreach_active(instance)");
            begin_scope();
            statement("ret[instance] = ", o, "_global(ptr[instance], value);");
            end_scope();
            statement("varying int vRet = *((varying int * uniform) &ret);");
            statement("return vRet;");
            end_scope();
            statement("");
        }
    }

    // Images
    // Currently implemented assumed atomic buffer counters. Probably needs work for non buffer based atomics
    {
        statement("");
        statement("//////////////////////////////");
        statement("// Image Load/Store/Size");
        statement("//////////////////////////////");
        statement("");
        statement("static SPIRV_INLINE varying float4 imageLoad(uniform image2D &image, varying int2 coord)");
        begin_scope();
        statement("const varying float r_255 = 1.0 / 255.0;");
        statement("varying float4 res = float4_init(0.0);");
        statement("if (coord.x >= image.width || coord.y >= image.height || coord.x < 0 || coord.y < 0)");
        statement("    return res;");
        statement("varying unsigned int index = coord.y * image.width + coord.x;");
        statement("varying pixel4D pix = image.data[index];");
        statement("res.x = ((float)pix.r) * r_255;");
        statement("res.y = ((float)pix.g) * r_255;");
        statement("res.z = ((float)pix.b) * r_255;");
        statement("res.w = ((float)pix.a) * r_255;");
        statement("return res;");
        end_scope();
        statement("");

        statement("");
        statement(
            "static SPIRV_INLINE void imageStore(uniform image2D &image, varying int2 coord, varying float4 rgba)");
        begin_scope();
        statement("if (coord.x >= image.width || coord.y >= image.height || coord.x < 0 || coord.y < 0)");
        statement("    return;");
        statement("varying float4 clamped_rgba = clamp(rgba, float4(0.0f), float4(1.0f));");
        statement("varying unsigned int index = coord.y * image.width + coord.x;");
        statement("varying pixel4D pix;");
        statement("pix.r = (unsigned int8)(clamped_rgba.x * 255);");
        statement("pix.g = (unsigned int8)(clamped_rgba.y * 255);");
        statement("pix.b = (unsigned int8)(clamped_rgba.z * 255);");
        statement("pix.a = (unsigned int8)(clamped_rgba.w * 255);");
        statement("image.data[index] = pix;");
        end_scope();
        statement("");

        statement("");
        statement("static SPIRV_INLINE uniform int2 imageSize(uniform image2D &image)");
        begin_scope();
        statement("uniform int2 ret = { image.width, image.height };");
        statement("return ret;");
        end_scope();
        statement("");
    }

    //
    // Unary Matrix ops
    //
    {

        vector<vector<string>> default_varying = {
            { "varying", "varying" },
            { "uniform", "uniform" },
        };

        codegen_matrix_transpose(3, default_varying);
        codegen_matrix_transpose(4, default_varying);
    }

    //
    // Binary Matrix ops
    //
    {

        vector<vector<string>> default_varying = {
            { "varying", "varying", "varying" },
        { "varying", "varying", "uniform" },
        { "varying", "uniform", "varying" },
        { "uniform", "uniform", "uniform" },
        };

        codegen_matrix_multiply(3, default_varying);
        codegen_matrix_multiply(4, default_varying);

        codegen_matrix_times_scalar(3, default_varying);
        codegen_matrix_times_scalar(4, default_varying);

        codegen_vector_times_matrix(3, default_varying);
        codegen_vector_times_matrix(4, default_varying);

        codegen_matrix_times_vector(3, default_varying);
        codegen_matrix_times_vector(4, default_varying);
    }

    // Inverse - requires some of the other functions to be defined
    {
        codegen_matrix_inverse(string("uniform"));
        codegen_matrix_inverse(string("varying"));
    }

}
